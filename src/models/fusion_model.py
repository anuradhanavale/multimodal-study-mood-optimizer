"""Late-fusion readiness model with modality-presence flags and modality dropout training."""
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from src.config import AUDIO_COLS, IMAGE_COLS, STRUCT_COLS
from src.preprocessing.text_processor import lexicon_features

GROUPS = ("struct", "text", "audio", "image")
OPTIONAL = ("text", "audio", "image")

class ReadinessModel:
    def __init__(self, groups=GROUPS, dropout=True, seed=0, svd_dim=12):
        self.groups, self.dropout, self.seed, self.svd_dim = tuple(groups), dropout, seed, svd_dim

    # ---- encoders -----------------------------------------------------
    def _ensure(self, df):
        df = df.copy()
        if "text" not in df:
            df["text"] = ""
        for c in STRUCT_COLS + AUDIO_COLS + IMAGE_COLS:
            if c not in df:
                df[c] = np.nan
        return df

    def _blocks(self, df):
        df = self._ensure(df)
        texts = df["text"].fillna("").astype(str)
        present = (texts.str.strip().str.len() > 0).to_numpy()
        tv = self.svd.transform(self.tfidf.transform(texts))
        lex = np.array([lexicon_features(t) for t in texts], float)
        tb = np.c_[tv, lex]
        tb[~present] = np.nan
        return {"struct": df[STRUCT_COLS].to_numpy(float), "text": tb,
                "audio": df[AUDIO_COLS].to_numpy(float), "image": df[IMAGE_COLS].to_numpy(float)}

    def _assemble(self, blocks, masks):
        parts = []
        for g in self.groups:
            b = blocks[g].copy()
            if g in masks:
                b[masks[g]] = np.nan
            pres = ~np.isnan(b).any(axis=1)
            parts.append(np.nan_to_num((b - self.mu[g]) / self.sd[g], nan=0.0))
            if g != "struct":
                parts.append(pres[:, None].astype(float))
        return np.hstack(parts)

    @staticmethod
    def _masks(n, drop):
        return {g: np.ones(n, bool) for g in drop}

    # ---- fit / predict --------------------------------------------------
    def fit(self, df, y):
        df = self._ensure(df).reset_index(drop=True)
        texts = df["text"].fillna("").astype(str)
        nonempty = texts[texts.str.strip().str.len() > 0]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(nonempty)
        k = max(2, min(self.svd_dim, len(self.tfidf.vocabulary_) - 1))
        self.svd = TruncatedSVD(k, random_state=self.seed).fit(self.tfidf.transform(nonempty))
        blocks = self._blocks(df)
        self.mu, self.sd = {}, {}
        for g in GROUPS:
            with np.errstate(all="ignore"):
                m, s = np.nanmean(blocks[g], 0), np.nanstd(blocks[g], 0)
            self.mu[g] = np.nan_to_num(m)
            s = np.nan_to_num(s, nan=1.0); s[s < 1e-6] = 1.0
            self.sd[g] = s
        n, rng = len(df), np.random.default_rng(self.seed)
        Xs, ys = [self._assemble(blocks, {})], [np.asarray(y)]
        if self.dropout:
            for _ in range(3):
                masks = {g: rng.random(n) < 0.35 for g in OPTIONAL if g in self.groups}
                Xs.append(self._assemble(blocks, masks)); ys.append(np.asarray(y))
        self.mlp = MLPClassifier((64, 32), alpha=1e-3, max_iter=400, early_stopping=True, random_state=self.seed)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mlp.fit(np.vstack(Xs), np.concatenate(ys))
        return self

    def predict_proba(self, df, drop=()):
        df = self._ensure(df).reset_index(drop=True)
        X = self._assemble(self._blocks(df), self._masks(len(df), drop))
        return self.mlp.predict_proba(X)

    def predict(self, df, drop=()):
        return self.mlp.classes_[self.predict_proba(df, drop).argmax(1)]

    # ---- explainability (feature-group ablation) --------------------------
    def present_groups(self, df):
        b = self._blocks(df)
        return [g for g in self.groups if not np.isnan(b[g][0]).any()]

    def explain(self, df):
        base = self.predict_proba(df)[0]
        k = int(base.argmax())
        out = {}
        for g in self.present_groups(df):
            out[g] = float(base[k] - self.predict_proba(df, drop=(g,))[0][k])
        return out
