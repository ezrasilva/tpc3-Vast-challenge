from src.data_loader import load_local_dataset
from src.preprocessing import preprocess_events
from src.analysis import file_based_posts, comparison_table, anomaly_funnel

raw, _ = load_local_dataset("data/VAST_Challenge_2026_MC2.zip")
df = preprocess_events(raw)
posts = file_based_posts(df)
assert len(posts) == 3
summary = comparison_table(df, posts)
assert set(summary["caso"]) == {"HiddenOrca", "MellowOtter", "SwiftWren"}
funnel = anomaly_funnel(df, posts)
assert int(funnel.iloc[-1]["quantidade"]) == 3
print("Smoke test concluído com sucesso.")
