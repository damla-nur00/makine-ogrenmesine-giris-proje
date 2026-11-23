"""PersonaFit denetimli yorgunluk modeli. CLI: egit veya tahmin."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pmdata import DATE, ID, TARGET, features, prepare

SEED = 42


def split_people(rows):
    people = np.sort(rows[ID].unique())
    if len(people) < 6:
        raise ValueError("Eğitim, doğrulama ve test için en az 6 etiketli katılımcı gerekli.")
    people = np.random.default_rng(SEED).permutation(people)
    count = max(2, round(len(people) * 0.125))
    return {"egitim": people[2 * count:], "dogrulama": people[count:2 * count],
            "test": people[:count]}


def models():
    yield "Sabit medyan", {}, DummyRegressor(strategy="median")
    for alpha in [1, 10, 50, 150]:
        yield "Ridge", {"alpha": alpha}, make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(), Ridge(alpha=alpha),
        )
    for depth in [4, 7, None]:
        for leaf in [5, 15, 30]:
            params = {"max_depth": depth, "min_samples_leaf": leaf, "max_features": 0.7}
            yield "Random Forest", params, make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **params),
            )


def bounded(estimator, x):
    return np.clip(estimator.predict(x), 1, 5)


def scores(rows, predicted):
    y = rows[TARGET]
    errors = pd.DataFrame({ID: rows[ID], "absolute": np.abs(y - predicted)})
    r2 = r2_score(y, predicted) if y.nunique() > 1 else None
    return {"MAE": mean_absolute_error(y, predicted),
            "RMSE": float(np.sqrt(mean_squared_error(y, predicted))),
            "R2": r2, "kisi_ortalama_MAE": errors.groupby(ID)["absolute"].mean().mean()}


def train(daily, output):
    x = features(daily)
    labeled = x[x[TARGET].between(1, 5)].copy()
    partitions = split_people(labeled)
    train_rows, val_rows, test_rows = [labeled[labeled[ID].isin(ids)] for ids in partitions.values()]
    candidates = {}
    validation = []
    for mode in ["sensor", "sensor_ve_gecmis_bildirim"]:
        columns = [col for col in train_rows if col not in [ID, DATE, TARGET]
                   and (mode != "sensor" or not col.startswith("fatigue_"))
                   and train_rows[col].notna().any()]
        if not columns:
            raise ValueError("Ölçüm sütunları boş.")
        for name, params, estimator in models():
            if name == "Sabit medyan" and mode != "sensor":
                continue
            estimator.fit(train_rows[columns], train_rows[TARGET])
            predicted = bounded(estimator, val_rows[columns])
            result = scores(val_rows, predicted)
            validation.append({"model": name, "girdiler": mode, "parametreler": params,
                               **result})
            if name not in candidates or result["MAE"] < candidates[name]["dogrulama"]["MAE"]:
                candidates[name] = {"model": estimator, "columns": columns, "girdiler": mode,
                                    "parametreler": params, "dogrulama": result}

    # Künye ana model olarak Random Forest'ı belirler. Test aşamasından önce sabitle.
    winner = candidates["Random Forest"]
    output.mkdir(parents=True, exist_ok=True)
    artifact = {"model": winner["model"], "columns": winner["columns"],
                "model_adi": "Random Forest", "girdiler": winner["girdiler"],
                "skor_araligi": [1, 5], "sabah_saati": "08:00"}
    joblib.dump(artifact, output / "personafit_yorgunluk.joblib", compress=3)

    summary = {"seed": SEED, "ana_model": artifact["model_adi"], "bolumler": {},
               "modeller": {}}
    for (name, ids), frame in zip(partitions.items(), (train_rows, val_rows, test_rows)):
        summary["bolumler"][name] = {"katilimcilar": ids.tolist(), "gozlem": len(frame)}
    predictions = test_rows[[ID, DATE, TARGET]].copy()
    for name, item in candidates.items():
        predicted = bounded(item["model"], test_rows[item["columns"]])
        predictions[name] = predicted
        summary["modeller"][name] = {
            "girdiler": item["girdiler"], "parametreler": item["parametreler"],
            "dogrulama": item["dogrulama"], "test": scores(test_rows, predicted),
            "kisi_bazinda_test": {
                str(person): {"n": len(group), "MAE": mean_absolute_error(group[TARGET], group[name])}
                for person, group in predictions.groupby(ID)
            },
        }
    (output / "metrikler.json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                           encoding="utf-8")
    pd.DataFrame(validation).sort_values("MAE").to_csv(output / "dogrulama_deneyleri.csv", index=False)
    return summary


def predict(daily, artifact, date=None):
    """Günlük DataFrame'den bir gün için kişi bazlı yorgunluk puanı üretir."""
    x = features(daily)
    if date is None:
        selected = x[DATE].eq(x.groupby(ID)[DATE].transform("max"))
    else:
        selected = x[DATE].eq(pd.Timestamp(date).normalize())
    x = x.loc[selected]
    if x.empty:
        raise ValueError("İstenen tarih için günlük satır bulunamadı.")
    inputs = x[artifact["columns"]]
    result = x[[ID, DATE]].copy()
    result["mevcut_ozellik"] = inputs.notna().sum(axis=1)
    result["tahmini_yorgunluk"] = np.nan
    usable = result["mevcut_ozellik"] > 0
    if usable.any():
        result.loc[usable, "tahmini_yorgunluk"] = bounded(artifact["model"], inputs.loc[usable])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("egit")
    fit.add_argument("--pmdata", type=Path, required=True)
    fit.add_argument("--cikti", type=Path, default=Path("sonuclar"))
    infer = commands.add_parser("tahmin")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--csv", type=Path, required=True)
    infer.add_argument("--tarih")
    args = parser.parse_args()
    if args.command == "egit":
        summary = train(prepare(args.pmdata), args.cikti)
        print(json.dumps(summary, indent=2))
    else:
        artifact = joblib.load(args.model)
        print(predict(pd.read_csv(args.csv), artifact, args.tarih).to_csv(index=False), end="")


if __name__ == "__main__":
    main()
