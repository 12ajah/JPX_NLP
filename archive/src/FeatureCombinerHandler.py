import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset as _Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn

class Dataset(_Dataset):
    def __init__(self, weekly_features, weekly_labels, max_sequence_length):
        # 共通する週のみを使うため、共通するindex情報を取得する
        mask_index = (
            weekly_features.index.get_level_values(0).unique() & weekly_labels.index
        )

        # 共通するindexのみのデータだけでreindexを行う。
        self.weekly_features = weekly_features[
            weekly_features.index.get_level_values(0).isin(mask_index)
        ]
        self.weekly_labels = weekly_labels.reindex(mask_index)
        
        # idからweekの情報を取得できるよう、id_to_weekをビルドする
        self.id_to_week = {
            id: week for id, week in enumerate(sorted(weekly_labels.index))
        }

        self.max_sequence_length = max_sequence_length

    def _shuffle_by_local_split(self, x, split_size=50):
        return torch.cat(
            [
                splitted[torch.randperm(splitted.size()[0])]
                for splitted in x.split(split_size, dim=0)
            ],
            dim=0,
        )

    def __len__(self):
        return len(self.weekly_labels)

    def __getitem__(self, id):
        # 付与されたidから週の情報を取得し、その週の情報から、特徴量とラベルを取得する。
        week = self.id_to_week[id]
        x = self.weekly_features.xs(week, axis=0, level=0)[-self.max_sequence_length :]
        y = self.weekly_labels.loc[week]

        # pytorchでは、データをtorch.Tensorタイプとして扱うことが要求される。
        # 全体的な特徴量(ニュースの情報)の順序は維持しつつ、入力とする特徴量を数分割し、その分割の中でシャッフルを行う。
        x = self._shuffle_by_local_split(torch.tensor(x.values, dtype=torch.float))
        y = torch.tensor(y, dtype=torch.float)

        # max_sequence_lengthに最大のsequenceを合わせ、sequenceがmax_sequence_lengthに達しない場合は、前から0を埋め、sequenceを合わせる
        if x.size()[0] < self.max_sequence_length:
            x = F.pad(x, pad=(0, 0, self.max_sequence_length - x.size()[0], 0))

        return x, y

class FeatureCombiner(nn.Module):
    def __init__(self, input_size, hidden_size, out_size=18, num_layers=2): # 768, 128
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTMの定義
        # batch_firstより、出力次元の最初がbatchとなる。
        # dropoutを用いて、内部状態のconnectionをdropすることより過学習を防ぐ。
        # Sequenceがかなり長く、入力の始めの方の情報の消失を防ぐため、bidirectionalのモデルを使う。
        self.cell = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )

        # より高次元の特徴量を抽出できるようにするため、classifierの手前で、compress_dim次元への線形圧縮を行う。
        self.compressor = nn.Linear(hidden_size * 2, hidden_size)

        # sentiment probabilityの出力層。
        self.classifier = nn.Linear(hidden_size, out_size)

        # outputの範囲を[0, 1]とする。
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 入力値xから出力までの流れを定義する。
        output, _ = self.cell(x)
        output = self.sigmoid(self.classifier(self.compressor(output[:, -1, :])))
        return output

    '''
    def extract_feature(self, x):
        # 入力値xから特徴量抽出までの流れを定義する。
        output, _ = self.cell(x)
        output = self.compressor(output[:, -1, :])
        return output
    '''

class FeatureCombinerHandler:
    def __init__(self, feature_combiner_params, store_dir):
        # モデル学習及び推論に用いるデバイスを定義する
        if torch.cuda.device_count() >= 1:
            self.device = 'cuda'
            print("[+] Set Device: GPU")
        else:
            self.device = 'cpu'
            print("[+] Set Device: CPU")

        # モデルのcheckpointや抽出した特徴量及びsentimentをstoreする場所を定義する。
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

        # 上記で作成したfeaturecombinerを定義する。
        self.feature_combiner = FeatureCombiner(**feature_combiner_params).to(
            self.device
        )

        # 学習に用いるoptimizerを定義する。
        self.optimizer = torch.optim.Adam(
            params=self.feature_combiner.parameters(), lr=0.001,
        )

        # ロス関数の定義
        self.criterion = nn.BCELoss().to(self.device)

        # モデルのcheck pointが存在する場合、モデルをロードする
        self._load_model()

    # 学習に必要なデータ(並列のためbatch化されたもの)をサンプルする。
    def _sample_xy(self, data_type):
        assert data_type in ("train", "val")

        # data_typeより、data_typeに合致したデータを取得するようにしている。
        if data_type == "train":
            # dataloaderをiteratorとして定義し、next関数として毎時のデータをサンプルすることができる。
            # Iteratorは全てのデータがサンプルされると、StopIterationのエラーを発するが、そのようなエラーが出たとき、
            # Iteratorを再定義し、データをサンプルするようにしている。
            try:
                x, y = next(self.iterable_train_dataloader)
            except StopIteration:
                self.iterable_train_dataloader = iter(self.train_dataloader)
                x, y = next(self.iterable_train_dataloader)

        elif data_type == "val":
            try:
                x, y = next(self.iterable_val_dataloader)
            except StopIteration:
                self.iterable_val_dataloader = iter(self.val_dataloader)
                x, y = next(self.iterable_val_dataloader)

        return x.to(self.device), y.to(self.device)

    # モデルのパラメータをアップデートするロジック
    def _update_params(self, loss):
        # ロスから、gradientを逆伝播し、パラメータをアップデートする
        loss.backward()
        self.optimizer.step()

    # 学習されたfeature_combinerのパラメータをcheck_pointとしてstoreするロジック
    def _save_model(self, epoch):
        torch.save(
            self.feature_combiner.state_dict(),
            os.path.join(self.store_dir, f"{epoch}.ckpt"),
        )
        print(f"[+] Epoch: {epoch}, Model is saved.")

    # 学習されたcheckpointが存在す場合、feature_combinerにそのパラメータをロードするロジック
    def _load_model(self):
        # cudaで学習されたモデルなどを、cpu環境下でロードするときはこのパラメータが必要となる。
        params_to_load = {}
        if self.device == "cpu":
            params_to_load["map_location"] = torch.device("cpu")

        # .ckptファイルを探し、古い順から新しい順にソートする。
        check_points = glob(os.path.join(self.store_dir, "*.ckpt"))
        check_points = sorted(
            check_points, key=lambda x: int(x.split("/")[-1].replace(".ckpt", "")),
        )

        # check_pointが存在しない場合は、スキップする。
        if len(check_points) == 0:
            print("[!] No exists checkpoint")
            return

        # 複数個のchieck_pointが存在する場合、一番最新のものを使い、モデルのパラメータをロードする
        check_point = check_points[-1]
        self.feature_combiner.load_state_dict(torch.load(check_point, **params_to_load))
        print("[+] Model is loaded")

    # Datasetからdataloaderを定義するロジック
    def _build_dataloader(
        self, dataloader_params, weekly_features, weekly_labels, max_sequence_length
    ):
        # 上記3で作成したしたdatasetを定義する
        dataset = Dataset(
            weekly_features=weekly_features,
            weekly_labels=weekly_labels,
            max_sequence_length=max_sequence_length,
        )

        # datasetのdataをiterableにロードできるよう、dataloaderを定義する、このとき、shuffle=Trueを渡すことで、データはランダムにサンプルされるようになる。
        return DataLoader(dataset=dataset, shuffle=True, **dataloader_params)

    # train用に、featuresとlabelsを渡し、datasetを定義し、dataloaderを定義するロジック
    def set_train_dataloader(
        self, dataloader_params, weekly_features, weekly_labels, max_sequence_length
    ):
        self.train_dataloader = self._build_dataloader(
            dataloader_params=dataloader_params,
            weekly_features=weekly_features,
            weekly_labels=weekly_labels,
            max_sequence_length=max_sequence_length,
        )

        # dataloaderからiteratorを定義する
        # iteratorはnext関数よりデータをサンプルすることが可能となる。
        self.iterable_train_dataloader = iter(self.train_dataloader)

    # validation用に、featuresとlabelsを渡し、datasetを定義し、dataloaderを定義するロジック
    def set_val_dataloader(
        self, dataloader_params, weekly_features, weekly_labels, max_sequence_length
    ):
        self.val_dataloader = self._build_dataloader(
            dataloader_params=dataloader_params,
            weekly_features=weekly_features,
            weekly_labels=weekly_labels,
            max_sequence_length=max_sequence_length,
        )

        # dataloaderからiteratorを定義する
        # iteratorはnext関数よりデータをサンプルすることが可能となる。
        self.iterable_val_dataloader = iter(self.val_dataloader)

    # 学習ロジック
    def train(self, n_epoch):
        best_loss = 1
        
        # n_epochの回数分、全学習データを複数回用いて学習する。
        for epoch in range(n_epoch):

            # 各々のepochごとのaverage lossを表示するため、lossをstoreするリストを定義する。
            train_losses = []
            test_losses = []

            # train_dataloaderの長さは、全ての学習データを一度用いるときの長さと同様である。
            # batchを組むと、その分train_dataloaderの長さは可変し、ちょうど一度全てのデータで学習できる長さを返す。
            for iter_ in tqdm(range(len(self.train_dataloader))):
                # パラメータをtrainableにするため、feature_combinerをtrainモードにする。
                self.feature_combiner.train()

                # trainデータをサンプルする。
                x, y = self._sample_xy(data_type="train")

                # feature_combinerに特徴量を入力し、sentiment scoreを取得する。
                preds = self.feature_combiner(x=x)

                # sentiment scoreとラベルとのロスを計算する。
                train_loss = self.criterion(preds, y)

                # 計算されたロスは、後ほどepochごとのdisplayに使用するため、storeしておく。
                train_losses.append(train_loss.detach().cpu())

                # lossから、gradientを逆伝播させ、パラメータをupdateする。
                self._update_params(loss=train_loss)

                # validation用のロースを計算する。
                # 毎回計算を行うとコストがかかってくるので、iter_毎5回ごとに計算を行う。
                if iter_ % 5 == 0:

                    # 学習を行わないため、feature_combinerをevalモードにしておく。
                    # evalモードでは、dropoutの影響を受けない。
                    self.feature_combiner.eval()

                    # 各パラメータごとのgradientを計算するとリソースが高まる。
                    # evaluationの時には、gradient情報を持たせないことで、メモリーの節約に繋がる。
                    with torch.no_grad():
                        # validationデータをサンプルする
                        x, y = self._sample_xy(data_type="val")

                        # feature_combinerに特徴量を入力し、sentiment scoreを取得する。
                        preds = self.feature_combiner(x=x)

                        # sentiment scoreとラベルとのロスを計算する。
                        test_loss = self.criterion(preds, y)

                        # 計算されたロスは、後ほどepochごとのdisplayに使用するため、storeしておく。
                        test_losses.append(test_loss.detach().cpu())

            # 毎epoch終了後、平均のロスをプリントする。
            print(
                f"epoch: {epoch}, train_loss: {np.mean(train_losses):.4f}, val_loss: {np.mean(test_losses):.4f}"
            )

            # 毎epoch終了後、モデルのパラメータをstoreする。
            if np.mean(train_losses) < best_loss:
              self._save_model(epoch=epoch)
              best_loss = np.mean(train_losses)

    # 特徴量から、合成特徴量を抽出するロジック
    def combine_features(self, features):
        # 学習を行わないため、feature_combinerをevalモードにしておく。
        self.feature_combiner.eval()

        # gradient情報を持たせないことで、メモリーの節約する。
        with torch.no_grad():

            # 特徴量をfeature_combinerのextract_feature関数に入力し、出力層手前の特徴量を抽出する。
            # 抽出するとき、tensorをcpu上に落とし、np.ndarray形式に変換する。
            return (
                self.feature_combiner.extract_feature(
                    x=torch.tensor(features, dtype=torch.float).to(self.device)
                )
                .cpu()
                .numpy()
            )

    # 特徴量から、翌週のsentimentを予測するロジック
    def predict_sentiment(self, features):
        # 学習を行わないため、feature_combinerをevalモードにしておく。
        self.feature_combiner.eval()

        # gradient情報を持たせないことで、メモリーの節約する。
        with torch.no_grad():

            # 特徴量をfeature_combinerに入力し、sentiment scoreを抽出する。
            # 抽出するとき、tensorをcpu上に落とし、np.ndarray形式に変換する。
            return (
                self.feature_combiner(x=torch.tensor(features, dtype=torch.float).to(self.device))
                .cpu()
                .numpy()
            )

    # weeklyグループされた特徴量を入力に、合成特徴量もしくは、sentiment scoreを抽出するロジック
    def generate_by_weekly_features(
        self, weekly_features, generate_target, max_sequence_length
    ):
        assert generate_target in ("features", "sentiment")
        generate_func = getattr(
            self,
            {"features": "combine_features", "sentiment": "predict_sentiment"}[
                generate_target
            ],
        )

        # グループごとに特徴量もしくは、sentiment scoreを抽出し、最終的に重ねて返すため、リストを作成する。
        outputs = []

        # ユニークな週indexを取得する。
        weeks = sorted(weekly_features.index.get_level_values(0).unique())

        for week in tqdm(weeks):
            # 各週ごとの特徴量を取得し、直近から、max_sequence_length分切る。
            features = weekly_features.xs(week, axis=0, level=0)[-max_sequence_length:]

            # 特徴量をモデルに入力し、合成特徴量もしくは、sentiment scoreを抽出し、outputsにappendする。
            # np.expand_dims(features, axis=0)を用いる理由は、特徴量合成機の入力期待値は、dimention0がbatchであるが、
            # featuresは、[1000, 768]の次元をもち、これらをunsqueezeし、[1, 1000, 768]に変換する必要がある。
            outputs.append(generate_func(features=np.expand_dims(features, axis=0)))

        # outputsを重ね、indexの情報とともにpd.DataFrame形式として返す。
        return pd.DataFrame(np.concatenate(outputs, axis=0), index=weeks)