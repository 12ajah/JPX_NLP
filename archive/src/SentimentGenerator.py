import math, os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import neologdn, unicodedata
import transformers
from transformers import BertJapaneseTokenizer
from FeatureCombinerHandler import FeatureCombinerHandler
import torch

class SentimentGenerator(object):
    article_columns = None
    device = None
    feature_extractor = None
    features = None
    target_article = None
    headline_feature_combiner_handler = None
    keywords_feature_combiner_handler = None
    punctuation_replace_dict = None
    punctuation_remove_list = None

    @classmethod
    def initialize(cls, base_dir="../model"):
        # 使用するcolumnをセットする。
        cls.article_columns = ["publish_datetime", "headline", "keywords", "company_g.stock_code"]

        # BERT特徴量抽出機をセットする。
        cls._set_device()
        cls._build_feature_extractor(base_dir)
        cls._build_tokenizer(base_dir)

        # LSTM特徴量合成機をセットする。
        cls.headline_feature_combiner_handler = FeatureCombinerHandler(
            feature_combiner_params={"input_size": 768, "hidden_size": 128},
            store_dir=f"{base_dir}/headline_features",
        )
        '''
        cls.keywords_feature_combiner_handler = FeatureCombinerHandler(
            feature_combiner_params={"input_size": 768, "hidden_size": 128},
            store_dir=f"{base_dir}/keywords_features",
        )
        '''

        # 置換すべき記号のdictionaryを作成する。
        JISx0208_replace_dict = {
            "髙": "高",
            "﨑": "崎",
            "濵": "浜",
            "賴": "頼",
            "瀨": "瀬",
            "德": "徳",
            "蓜": "配",
            "昻": "昂",
            "桒": "桑",
            "栁": "柳",
            "犾": "犹",
            "琪": "棋",
            "裵": "裴",
            "魲": "鱸",
            "羽": "羽",
            "焏": "丞",
            "祥": "祥",
            "曻": "昇",
            "敎": "教",
            "澈": "徹",
            "曺": "曹",
            "黑": "黒",
            "塚": "塚",
            "閒": "間",
            "彅": "薙",
            "匤": "匡",
            "冝": "宜",
            "埇": "甬",
            "鮏": "鮭",
            "伹": "但",
            "杦": "杉",
            "罇": "樽",
            "柀": "披",
            "﨤": "返",
            "寬": "寛",
            "神": "神",
            "福": "福",
            "礼": "礼",
            "贒": "賢",
            "逸": "逸",
            "隆": "隆",
            "靑": "青",
            "飯": "飯",
            "飼": "飼",
            "緖": "緒",
            "埈": "峻",
        }

        cls.punctuation_replace_dict = {
            **JISx0208_replace_dict,
            "《": "〈",
            "》": "〉",
            "『": "「",
            "』": "」",
            "“": '"',
            "!!": "!",
            "〔": "[",
            "〕": "]",
            "χ": "x",
        }

        # 取り除く記号リスト。
        cls.punctuation_remove_list = [
            "|",
            "■",
            "◆",
            "●",
            "★",
            "☆",
            "♪",
            "〃",
            "△",
            "○",
            "□",
        ]

    @classmethod
    def _set_device(cls):
        # 使用可能なgpuがある場合、そちらを利用し特徴量抽出を行う
        if torch.cuda.device_count() >= 1:
            cls.device = "cuda"
            print("[+] Set Device: GPU")
        else:
            cls.device = "cpu"
            print("[+] Set Device: CPU")
    
    @classmethod
    def load_feature_extractor(cls, model_dir, download=False, save_local=False):
        # 特徴量抽出のため事前学習済みBERTモデルを用いる。
        # ここでは、"cl-tohoku/bert-base-japanese-whole-word-masking"モデルを使用しているが、異なる日本語BERTモデルを用いても良い。
        target_model = "cl-tohoku/bert-base-japanese-whole-word-masking"
        save_dir = os.path.abspath(
            f"{model_dir}/transformers_pretrained/{target_model}"
        )
        if download:
            pretrained_model = target_model
        else:
            pretrained_model = save_dir

        feature_extractor = transformers.BertModel.from_pretrained(
            pretrained_model,
            return_dict=True,
            output_hidden_states=True,
        )

        if download and save_local:
            print(f"[+] save feature_extractor: {save_dir}")
            feature_extractor.save_pretrained(save_dir)

        return feature_extractor

    @classmethod
    def _build_feature_extractor(cls, model_dir, download=False):
        # 事前学習済みモデルを取得
        cls.feature_extractor = cls.load_feature_extractor(model_dir, download)

        # 使用するdeviceを指定
        cls.feature_extractor = cls.feature_extractor.to(cls.device)

        # 今回、学習は行わない。特徴量抽出のためなので、評価モードにセットする。
        cls.feature_extractor.eval()

        print("[+] Built feature extractor")
    
    @classmethod
    def load_bert_tokenizer(cls, model_dir, download=False, save_local=False):
        # BERTモデルの入力とするコーパスはそのBERTモデルが学習された時と同様の前処理を行う必要がある。
        # 今回使用する"cl-tohoku/bert-base-japanese-whole-word-masking"モデルは、
        # mecab-ipadic-NEologdによりトークナイズされ、その後Wordpiece subword encoderよりsubword化している。
        # Subwordとは形態素の類似な概念として、単語をより小さい意味のある単位に変換したものである。
        # transformersのBertJapaneseTokenizerは、その事前学習モデルの学習時と同様の前処理を簡単に使用することができる。
        # この章ではBertJapaneseTokenizerを利用し、トークナイズ及びsubword化を行う。
        target_model = "cl-tohoku/bert-base-japanese-whole-word-masking"
        save_dir = os.path.abspath(
            f"{model_dir}/transformers_pretrained/{target_model}"
        )
        if download:
            pretrained_model = target_model
        else:
            pretrained_model = save_dir
        bert_tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)
        if download and save_local:
            print(f"[+] save bert_tokenizer: {save_dir}")
            bert_tokenizer.save_pretrained(save_dir)

        return bert_tokenizer

    @classmethod
    def _build_tokenizer(cls, model_dir, download=False):
        # トークナイザーを取得
        cls.bert_tokenizer = cls.load_bert_tokenizer(model_dir, download)

        print("[+] Built bert tokenizer")

    @classmethod
    def load_articles(cls, path, start_dt=None, end_dt=None):
        # csvをロードする
        # headline、keywordsをcolumnとして使用。publish_datetimeをindexとして使用。
        articles = pd.read_csv(path)[cls.article_columns].set_index("publish_datetime")

        # 列名を変更
        articles.rename(columns={'company_g.stock_code':'Local Code'}, inplace=True)

        # str形式のdatetimeをpd.Timestamp形式に変換
        articles.index = pd.to_datetime(articles.index)

        # NaN値を取り除く
        articles = articles.dropna(subset=["headline", "keywords"])

        # 必要な場合、使用するデータの範囲を指定する
        return articles[start_dt:end_dt]

    @classmethod
    def normalize_articles(cls, articles):
        articles = articles.copy()

        # 欠損値を取り除く
        articles = articles.dropna(subset=["headline", "keywords"])

        for column in articles.columns[:1]:
            # スペース(全角スペースを含む)はneologdn正規化時に全て除去される。
            # ここでは、スペースの情報が失われないように、スペースを全て改行に書き換え、正規化後スペースに再変換する。
            articles[column] = articles[column].apply(lambda x: "\n".join(x.split()))

            # neologdnを使って正規化を行う。
            articles[column] = articles[column].apply(lambda x: neologdn.normalize(x))

            # 改行をスペースに置換する。
            articles[column] = articles[column].str.replace("\n", " ")

        return articles

    @classmethod
    def handle_punctuations_in_articles(cls, articles):
        articles = articles.copy()

        for column in articles.columns[:1]:
            # punctuation_remove_listに含まれる記号を除去する
            articles[column] = articles[column].str.replace(
                fr"[{''.join(cls.punctuation_remove_list)}]", ""
            )

            # punctuation_replace_dictに含まれる記号を置換する
            for replace_base, replace_target in cls.punctuation_replace_dict.items():
                articles[column] = articles[column].str.replace(
                    replace_base, replace_target
                )

            # unicode正規化を行う
            articles[column] = articles[column].apply(
                lambda x: unicodedata.normalize("NFKC", x)
            )

        return articles

    @classmethod
    def drop_remove_list_words(cls, articles, remove_list_words=["人事"]):
        articles = articles.copy()

        for remove_list_word in remove_list_words:
            # headlineもしくは、keywordsどちらかでremove_list_wordを含むニュース記事のindexマスクを作成。
            drop_mask = articles["headline"].str.contains(remove_list_word) | articles[
                "keywords"
            ].str.contains(remove_list_word)

            # remove_list_wordを含まないニュースだけに精製する。
            articles = articles[~drop_mask]

        return articles

    @classmethod
    def build_inputs(cls, texts, max_length=512):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for text in texts:
            encoded = cls.bert_tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
            )

            input_ids.append(encoded["input_ids"])
            token_type_ids.append(encoded["token_type_ids"])
            attention_mask.append(encoded["attention_mask"])

        # torchモデルに入力するためにはtensor形式に変え、deviceを指定する必要がある。
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(cls.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(cls.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(cls.device)

        return input_ids, token_type_ids, attention_mask

    @classmethod
    def generate_features(cls, input_ids, token_type_ids, attention_mask):
        output = cls.feature_extractor(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        features = output["hidden_states"][-2].mean(dim=1).cpu().detach().numpy()

        return features

    @classmethod
    def generate_features_by_texts(cls, texts, batch_size=2, max_length=512):
        n_batch = math.ceil(len(texts) / batch_size)

        features = []
        for idx in tqdm(range(n_batch)):
            input_ids, token_type_ids, attention_mask = cls.build_inputs(
                texts=texts[batch_size * idx : batch_size * (idx + 1)],
                max_length=max_length,
            )

            features.append(
                cls.generate_features(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
            )

        features = np.concatenate(features, axis=0)

        # 抽出した特徴量はnp.ndarray形式となっており、これらは、日付の情報を失っているため、pd.DataFrame形式に変換する。
        return pd.DataFrame(features, index=texts.index)

    @classmethod
    def _build_weekly_group(cls, df):
        # index情報から、(year, week)の情報を得る。
        return pd.Series(list(zip(df.index.year, df.index.week)), index=df.index)

    @classmethod
    def build_weekly_features(cls, features, boundary_week):
        assert isinstance(boundary_week, tuple)

        weekly_group = cls._build_weekly_group(df=features)
        features = features.groupby(weekly_group).apply(lambda x: x[:])

        train_features = features[features.index.get_level_values(0) <= boundary_week]
        test_features = features[features.index.get_level_values(0) > boundary_week]

        return {"train": train_features, "test": test_features}

    @classmethod
    def generate_lstm_features(
        cls,
        article_path,
        start_dt=None,
        boundary_week=(2020, 26),
        target_feature_types=None,
    ):
        # target_feature_typesが指定されなかったらデフォルト値設定
        dfault_target_feature_types = [
            "headline",
            "keywords",
        ]
        if target_feature_types is None:
            target_feature_types = dfault_target_feature_types
        # feature typeが想定通りであることを確認
        assert set(target_feature_types).issubset(dfault_target_feature_types)

        # ニュースデータをロードする。
        articles = cls.load_articles(start_dt=start_dt, path=article_path)

        # 前処理を行う。
        articles = cls.normalize_articles(articles)
        articles = cls.handle_punctuations_in_articles(articles)
        articles = cls.drop_remove_list_words(articles)

        # 前処理後のarticlesを保存しておく
        cls.target_article = articles.copy()

        # headlineとkeywordsの特徴量をdict型で返す。
        lstm_features = {}

        for feature_type in target_feature_types:
            # コーパス全体のBERT特徴量を抽出する。
            features = cls.generate_features_by_texts(texts=articles[feature_type])

            # feature_typeに合致するfeature_combiner_handlerをclsから取得する。
            feature_combiner_handler = {
                "headline": cls.headline_feature_combiner_handler,
                "keywords": cls.keywords_feature_combiner_handler,
            }[feature_type]

            # 特徴量を週毎のグループ化する。
            weekly_features = cls.build_weekly_features(features, boundary_week)["test"]

             # BERT特徴量を保存しておく
            if cls.features == None:
                cls.features = {}
            cls.features[feature_type] = weekly_features.copy()

            # Sentiment scoreを抽出する。
            lstm_features[
                f"{feature_type}_features"
            ] = feature_combiner_handler.generate_by_weekly_features(
                weekly_features=weekly_features,
                generate_target="sentiment",
                max_sequence_length=10000,
            )

        return lstm_features
