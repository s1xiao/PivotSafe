import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def _to_spark_uri(path_str: str) -> str:
    """
    Spark 在配置了 fs.defaultFS=hdfs 时，裸绝对路径会被当成 HDFS 路径。
    这里显式转成 file:// URI，强制按本地文件系统访问。
    """
    s = (path_str or "").strip()
    if "://" in s:
        return s
    return Path(s).resolve().as_uri()


def main() -> None:
    parser = argparse.ArgumentParser(description="CCCC batch post-process with Spark")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--modality", required=True)
    args = parser.parse_args()

    input_jsonl_local = Path(args.input_jsonl).resolve()
    output_dir_local = Path(args.output_dir).resolve()
    output_dir_local.mkdir(parents=True, exist_ok=True)

    input_jsonl_uri = _to_spark_uri(str(input_jsonl_local))
    parquet_out_uri = _to_spark_uri(str(output_dir_local / "parquet"))
    csv_out_uri = _to_spark_uri(str(output_dir_local / "csv"))

    spark = SparkSession.builder.appName(f"cccc_batch_post_{args.job_id}").getOrCreate()
    try:
        df = spark.read.json(input_jsonl_uri)

        for c, t in [
            ("file_name", "string"),
            ("status", "string"),
            ("label_pred", "string"),
            ("ai_score", "double"),
            ("message", "string"),
        ]:
            if c not in df.columns:
                df = df.withColumn(c, F.lit(None).cast(t))

        success_df = df.filter(F.col("status") == F.lit("success"))
        total = df.count()
        success = success_df.count()
        failed = total - success
        ai_files = success_df.filter(F.lower(F.col("label_pred")) == F.lit("ai")).count()
        human_files = success - ai_files
        avg_row = success_df.agg(F.avg(F.col("ai_score")).alias("avg_ai_score")).collect()[0]
        avg_ai_score = float(avg_row["avg_ai_score"]) if avg_row["avg_ai_score"] is not None else 0.0

        # 结果落盘：Parquet + CSV，便于后续 Hive / BI 对接
        df.coalesce(1).write.mode("overwrite").parquet(parquet_out_uri)
        (
            df.select("file_name", "status", "label_pred", "ai_score", "message")
            .coalesce(1)
            .write.mode("overwrite")
            .option("header", True)
            .csv(csv_out_uri)
        )

        summary = {
            "job_id": args.job_id,
            "modality": args.modality,
            "total_files": int(total),
            "success_files": int(success),
            "failed_files": int(failed),
            "ai_files": int(ai_files),
            "human_files": int(human_files),
            "avg_ai_score": round(avg_ai_score, 4),
        }
        with (output_dir_local / "spark_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
