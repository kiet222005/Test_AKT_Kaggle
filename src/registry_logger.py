from pathlib import Path
import pandas as pd


class ResultsRegistry:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent

        self.registry_path = base_dir / "results" / "registry.csv"

        # Tạo thư mục results nếu chưa có
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        config,
        metrics,
        stopped_epoch,
        time_per_epoch,
        raw_pred_path,
        author=None,
        **kwargs,
    ):
        if author is None:
            author = config.get("experiment", {}).get("author", "N/A")

        row = {
            "experiment": config["experiment"]["name"],
            "author": author,
            "model": config["model"]["name"],
            "fold": config["training"]["fold"],
            "val_auc": metrics["val_auc"],
            "val_acc": metrics["val_acc"],
            "test_auc": metrics["test_auc"],
            "test_acc": metrics["test_acc"],
            "test_rmse": metrics["test_rmse"],
            "stopped_epoch": stopped_epoch,
            "time_per_epoch": round(time_per_epoch, 2),
            "raw_pred_path": raw_pred_path,
        }

        new_row = pd.DataFrame([row])

        if self.registry_path.exists():
            df = pd.read_csv(self.registry_path)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(self.registry_path, index=False)

        print(f"[Registry] Saved: {self.registry_path}")