import numpy as np
import pandas as pd
from scipy.stats import zscore

import sys, os, pickle, io
from SentimentGenerator import SentimentGenerator

class ScoringService(object):
    # テスト期間開始日
    TEST_START = "2021-02-01"
    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None
    # センチメントの分布をこの変数に読み込む
    df_sentiment_dist = None

    @classmethod
    def get_inputs(cls, dataset_dir='../dataset'):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            #"stock_price": f"{dataset_dir}/stock_price.csv.gz",
            #"stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            #"stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            # ニュースデータ
            "tdnet": f"{dataset_dir}/tdnet.csv.gz",
            #"disclosureItems": f"{dataset_dir}/disclosureItems.csv.gz",
            "nikkei_article": f"{dataset_dir}/nikkei_article.csv.gz",
            #"article": f"{dataset_dir}/article.csv.gz",
            #"industry": f"{dataset_dir}/industry.csv.gz",
            #"industry2": f"{dataset_dir}/industry2.csv.gz",
            #"region": f"{dataset_dir}/region.csv.gz",
            #"theme": f"{dataset_dir}/theme.csv.gz",
            # 目的変数データ
            #"stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs, load_data):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            # 必要なデータのみ読み込みます
            if k not in load_data:
                continue
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs
    
    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["universe_comp2"] == True][
            "Local Code"
        ].values
        return cls.codes
    
    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        m = os.path.join(model_path, "kmeans_model.pickle")
        with open(m, "rb") as f:
            # pickle形式で保存されているモデルを読み込み
            cls.models['kmeans'] = pickle.load(f)
        
        # SentimentGeneratorクラスの初期設定を実施
        SentimentGenerator.initialize(model_path)
        
        # 事前に計算済みのセンチメントを分布として使用するために読み込みます
        cls.df_sentiment_dist = cls.load_sentiments(
            f"{model_path}/headline_features/LSTM_sentiment.pkl"
        )

        return True
    
    @classmethod
    def transform_yearweek_to_monday(cls, year, week):
        """
        ニュースから抽出した特徴量データのindexは (year, week) なので、
        (year, week) => YYYY-MM-DD 形式(月曜日) に変換します。
        """
        for s in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            if s.week == week:
                # to return Monday of the first week of the year
                # e.g. "2020-01-01" => "2019-12-30"
                return s - pd.Timedelta(f"{s.dayofweek}D")
    
    @classmethod
    def load_sentiments(cls, path=None):
        #DIST_END_DT = "2020-09-25"

        print(f"[+] load prepared sentiment: {path}")

        # 事前に出力したセンチメントの分布を読み込み
        df_sentiments = pd.read_pickle(path)

        # indexを日付型に変換します変換します。
        df_sentiments.loc[:, "index"] = df_sentiments.index.map(
            lambda x: cls.transform_yearweek_to_monday(x[0], x[1])
        )
        # indexを設定します
        df_sentiments.set_index("index", inplace=True)
        
        # 金曜日日付に変更します
        df_sentiments.index = df_sentiments.index + pd.Timedelta("4D")

        return df_sentiments
    
    @classmethod
    def get_sentiment(cls, inputs, start_dt="2020-12-31"):
        # ニュース見出しデータへのパスを指定
        article_path = inputs["nikkei_article"]
        target_feature_types = ["headline"]
        df_sentiments = SentimentGenerator.generate_lstm_features(
            article_path,
            start_dt=start_dt,
            target_feature_types=target_feature_types,
        )["headline_features"]

        df_sentiments.loc[:, "index"] = df_sentiments.index.map(
            lambda x: cls.transform_yearweek_to_monday(x[0], x[1])
        )
        df_sentiments.set_index("index", inplace=True)
        #df_sentiments.rename(columns={0: "headline_m2_sentiment_0"}, inplace=True)
        return df_sentiments
    
    @classmethod
    def get_cluster(cls, dfs, inputs):
        # ニュースとそのBERT特徴量を取り出す & 結合
        articles = SentimentGenerator.target_article
        weekly_group = SentimentGenerator._build_weekly_group(articles)
        articles = articles.groupby(weekly_group).apply(lambda x: x[:])
        features = SentimentGenerator.features['headline']
        features.columns = [f'feature_{i}' for i in features.columns]

        # ニュースと業種を紐づける
        articles = pd.concat([articles, features], axis=1).copy()
        articles.dropna(subset=['Local Code'], inplace=True)

        # K-means
        predict = cls.models['kmeans'].predict(articles.loc[:, articles.columns.str.contains('feature_')])
        articles['cluster'] = predict

        return articles[['Local Code', 'cluster']]

    @classmethod
    def get_exclude(
        cls,
        df_tdnet,  # tdnetのデータ
        start_dt=None,  # データ取得対象の開始日、Noneの場合は制限なし
        end_dt=None,  # データ取得対象の終了日、Noneの場合は制限なし
        lookback=7,  # 除外考慮期間 (days)
        target_day_of_week=4,  # 起点となる曜日
    ):
        # 特別損失のレコードを取得
        special_loss = df_tdnet[df_tdnet["disclosureItems"].str.contains('201"')].copy()
        # 日付型を調整
        special_loss["date"] = pd.to_datetime(special_loss["disclosedDate"])
        # 処理対象開始日が設定されていない場合はデータの最初の日付を取得
        if start_dt is None:
            start_dt = special_loss["date"].iloc[0]
        # 処理対象終了日が設定されていない場合はデータの最後の日付を取得
        if end_dt is None:
            end_dt = special_loss["date"].iloc[-1]
        #  処理対象日で絞り込み
        special_loss = special_loss[
            (start_dt <= special_loss["date"]) & (special_loss["date"] <= end_dt)
        ]
        # 出力用にカラムを調整
        res = special_loss[["code", "disclosedDate", "date"]].copy()
        # 銘柄コードを4桁にする
        res["code"] = res["code"].astype(str).str[:-1]
        # 予測の基準となる金曜日の日付にするために調整
        res["remain"] = (target_day_of_week - res["date"].dt.dayofweek) % 7
        res["start_dt"] = res["date"] + pd.to_timedelta(res["remain"], unit="d")
        res["end_dt"] = res["start_dt"] + pd.Timedelta(days=lookback)
        # 出力するカラムを指定
        columns = ["code", "date", "start_dt", "end_dt"]
        return res[columns].reset_index(drop=True)

    @classmethod
    def strategy(cls, df_sentiments, df_cluster):
        df_target = df_cluster.copy()
        index_lev1 = sorted(list(set(df_target.index.get_level_values(0)))) # 念のため並び替えをしておく

        # 基準化
        df_sentiments = (df_sentiments - cls.df_sentiment_dist.mean()) / cls.df_sentiment_dist.std()
      
        for i, ind in enumerate(index_lev1):
            # 週次の投資対象一覧を取得
            df_weekly_target = df_target.loc[ind].copy()

            ###################
            # クラスター番号によるスクリーニング
            ###################
            # クラスター7が5銘柄より少ない場合は、クラスター4だけ除外する
            if  len(df_weekly_target[df_weekly_target['cluster'] == 7]['Local Code'].unique()) < 5:
                df_weekly_target = df_weekly_target[df_weekly_target['cluster'] != 4].copy()
            else:
                df_weekly_target = df_weekly_target[df_weekly_target['cluster'] == 7].copy()
            # 銘柄数が足りない場合は、クラスター番号によるスクリーニングをしない
            if len(df_weekly_target) < 5:
                df_weekly_target = df_target.loc[ind]

            ###################
            # 業種区分によるスクリーニング
            ###################
            df_target_tmp = df_weekly_target.copy()
            del_sector = [2, 11, 12, 15]
            df_weekly_target = df_weekly_target[~df_weekly_target['sector'].isin(del_sector)].copy()
            if len(df_weekly_target['Local Code'].unique()) < 5:
                df_weekly_target = df_target.copy()

            ###################
            #  特別損失銘柄の除外
            ###################
            df_target_tmp = df_weekly_target.copy()
            df_exclude = cls.get_exclude(cls.dfs['tdnet'])
            # 除外用にユニークな列を作成します。
            df_exclude.loc[:, "date-code_lastweek"] = (
                df_exclude.loc[:, "start_dt"].dt.strftime("%Y-%m-%d-")
                + df_exclude.loc[:, "code"]
            )
            df_exclude.loc[:, "date-code_thisweek"] = (
                df_exclude.loc[:, "end_dt"].dt.strftime("%Y-%m-%d-")
                + df_exclude.loc[:, "code"]
            )
            #
            df_weekly_target.loc[:, "date-code_lastweek"] = (df_weekly_target.index - pd.Timedelta("7D")).strftime(
                "%Y-%m-%d-"
            ) + df_weekly_target.loc[:, "Local Code"].astype(str)
            df_weekly_target.loc[:, "date-code_thisweek"] = df_weekly_target.index.strftime("%Y-%m-%d-") + df_weekly_target.loc[
                :, "Local Code"
            ].astype(str)
            # 特別損失銘柄を除外
            df_weekly_target = df_weekly_target.loc[
                ~df_weekly_target.loc[:, "date-code_lastweek"].isin(
                    df_exclude.loc[:, "date-code_lastweek"]
                )
            ]
            df_weekly_target = df_weekly_target.loc[
                ~df_weekly_target.loc[:, "date-code_thisweek"].isin(
                    df_exclude.loc[:, "date-code_thisweek"]
                )
            ]
            if len(df_weekly_target['Local Code'].unique()) < 5:
                df_weekly_target = df_target.copy()

            ###################
            # 各業種への投資比率
            ###################
            # セクターへの投資比率をセンチメントスコアから決定する
            weekly_sentiment = df_sentiments.iloc[i, df_weekly_target['sector'].unique().tolist()] + 1 # +1することでマイナスを解消(ウェイト計算のため)
            weekly_sentiment[weekly_sentiment < 0]  = 0 # 稀にマイナスの場合があるため、0にしておく
            sector_weights = weekly_sentiment / weekly_sentiment.sum()
            # 各セクターの銘柄数を算出し、各セクターの銘柄投資比率(均等)を決定する
            stock_num = df_weekly_target.groupby('sector').count()['Local Code']
            df_weekly_target['weight'] = df_weekly_target['sector'].apply(lambda x: sector_weights[x] / stock_num[x])

            ###################
            # 現金比率
            ###################
            invest_total = 1000000 # 100万円

            # 投資対象ユニバースの予測スコアから現金比率を決定する
            weekly_market_sentiment = df_sentiments.iloc[i, 0]
            z = (cls.df_sentiment_dist[0] - cls.df_sentiment_dist.mean()[0]) / cls.df_sentiment_dist.std()[0] 
            p = np.percentile(z, [25, 50, 75])
            tile = np.digitize(weekly_market_sentiment, p) # 0～3の値:値が高いほど、投資比率を高くする
            # (メモ)
            # 投資比率は適当(0.5基準の10%刻み)。現金資産もあったほうが良いとも割れるため、少なくとも20%は充てるようにした。
            if tile == 0:
                invest_total = 0.5 * invest_total # 50%現金
            elif tile == 1:
                invest_total = 0.6 * invest_total # 60%現金
            elif tile == 2:
                invest_total = 0.7 * invest_total # 70%現金
            else:
                invest_total = 0.8 * invest_total # 80％投資

            df_weekly_target['budget'] = df_weekly_target['weight'] * invest_total

            # 直近のニュースを優先とする(ウェイトの調整は諦め、ソートで対応)
            df_weekly_target.sort_index(ascending=False, inplace=True)

            # 日付を(Y, W)を戻す
            df_weekly_target.reset_index(inplace=True)
            df_weekly_target.index = pd.MultiIndex.from_tuples([ind] * len(df_weekly_target)) 

            # 保存
            if i == 0:
                df = df_weekly_target.copy()
            else:
                df = pd.concat([df, df_weekly_target], axis=0)
      
        return df

    @classmethod
    def predict(
        cls,
        inputs,
        start_dt=TEST_START,
        load_data=["stock_list", 
                   "tdnet", 
                   "purchase_date",
                   ],
    ):
        """Predict method
        Args:
            inputs (dict[str]): paths to the dataset files
            codes (list[int]): traget codes
            start_dt (str): specify target purchase date
            load_data (list[str]): list of data to load
        Returns:
            str: Inference for the given input.
        """
        # データ読み込み
        if cls.dfs is None:
            print("[+] load data")
            cls.get_dataset(inputs, load_data)
            cls.get_codes(cls.dfs)

        # purchase_date が存在する場合は予測対象日を上書き
        if "purchase_date" in cls.dfs.keys():
            # purchase_dateの最も古い日付を設定
            start_dt = cls.dfs["purchase_date"].sort_values("Purchase Date").iloc[0, 0]

        # 日付型に変換
        start_dt = pd.Timestamp(start_dt)
        # 予測対象日の月曜日日付が指定されているため
        # 特徴量の抽出に使用する1週間前の日付に変換します
        start_dt -= pd.Timedelta("7D")
        # 文字列型に戻す
        start_dt = start_dt.strftime("%Y-%m-%d")

        ###################
        # センチメント情報取得
        ###################
        # ニュース見出しデータへのパスを指定
        df_sentiments = cls.get_sentiment(inputs, start_dt=start_dt)
        #
        # 金曜日日付に変更
        df_sentiments.index = df_sentiments.index + pd.Timedelta("4D")
        
        ###################
        # K-means(クラスタリング)
        ###################
        # 各ニュースにクラスター番号を付ける
        df_cluster = cls.get_cluster(cls.dfs, inputs)
        #
        # 業種区分を紐づける(ニュースが複数銘柄のものは欠落する)←欠落しないように対応すべきだったのかも
        df_sector = cls.dfs['stock_list'][['Local Code', '17 Sector(Code)']]
        df_cluster['sector'] = df_cluster['Local Code'].apply(lambda x: df_sector[df_sector['Local Code'].astype(str) == str(x)]['17 Sector(Code)'].values)
        df_cluster['sector'] = df_cluster['sector'].apply(lambda x: x[0] if len(x) != 0 else np.nan) # 業種区分が無い場合、[]となるためnanに変換(力技過ぎる...)
        df_cluster.dropna(subset=['sector'], inplace=True)

        ###################
        # 銘柄選定
        ###################
        df = cls.strategy(df_sentiments, df_cluster)
        
        # 結果を以下のcsv形式で出力する
        # 1列目:date
        # 2列目:Local Code
        # 3列目:budget
        # headerあり、2列目3列目はint64

        # 月曜日日付(ニュース公表の週初)に変更後、1週間ずらす
        df.index = df.index.map(lambda x: cls.transform_yearweek_to_monday(x[0], x[1]))
        df.index = df.index + pd.offsets.Week(1)

        # 出力用に調整
        df.index.name = "date"
        df.reset_index(inplace=True)
        df['Local Code'] = df['Local Code'].astype(int)
        df['budget'] = df['budget'].astype(int)

        # 出力対象列を定義
        output_columns = ["date", "Local Code", "budget"]

        out = io.StringIO()
        df.to_csv(out, header=True, index=False, columns=output_columns)

        # csvで保存しておく
        df[output_columns].to_csv('./result.csv')

        return out.getvalue()