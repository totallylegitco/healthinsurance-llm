from .ca_data_utils import *
import re
from .utils import *
import multiprocessing
from typing import Tuple, Optional, Iterator
import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import udf, pandas_udf, struct, col
from pyspark.sql.types import *


spark = SparkSession \
  .builder \
  .appName("CA Data") \
  .getOrCreate()

# Load the IMR data
imr_df = spark.read.csv(path=imr_data_input_path, header=True).select(*relevant_columns)
filtered_imr_records = imr_df.filter(imr_df["Determination"].contains("Overturned"))
filtered_imr_records.cache()
overturned_count = filtered_imr_records.count()
print(f"Processing a total of {overturned_count}")
filtered_imr_records.show(
)

@udf
def get_treatment_from_findings_and_categories_udf(
        findings: Optional[str], tc: Optional[str], tsc: Optional[str]) -> Optional[str]:
    return get_treatment_from_findings_and_categories(findings, tc, tsc)
    

@udf
def get_diagnosis_from_findings_and_categories_udf(
        findings: Optional[str], dc: Optional[str], dsc: Optional[str]) -> Optional[str]:
    return get_diagnosis_from_findings_and_categories(findings, dc, dsc)

def generate_prompts_from_imr_df(imrs_itr: Iterator[pd.DataFrame]):
    for imrs in imrs_itr:
        result = imrs.apply(generate_prompts_from_imr, axis=1, result_type="expand")
        pd.set_option('display.max_columns', None)
        result.rename(columns={'0': 'index', '1': 'prompts', '2': 'known'}, inplace=True)
        print(result)
        print(result.info())
        yield result

# Add what we _suspect_ the treatment is
enriched_imr_records = filtered_imr_records.withColumn(
    "treatment_extracted",
    get_treatment_from_findings_and_categories_udf(
        "Findings",
        "TreatmentCategory",
        "TreatmentSubCategory")
).withColumn(
    "diagnosis_extracted",
    get_treatment_from_findings_and_categories_udf(
        "Findings",
        "TreatmentCategory",
        "TreatmentSubCategory")
)
enriched_imr_records.cache()
enriched_imr_records.show()

schema = StructType([
    StructField("index", StringType(), False),
    StructField("prompts", MapType(StringType(), ArrayType(StringType())), False),
    StructField("known", MapType(StringType(), StringType()), False)
])
prompts_df = enriched_imr_records.mapInPandas(generate_prompts_from_imr_df, schema)
prompts_df.cache()
prompts_df.show()
prompts_df.count()
enriched_imr_records.unpersist()
filtered_imr_records.unpersist()
# Optional: load existing records
#existing_records = spark.sparkContext.wholeTextFiles("./generated-llm-data")
#existing_records_df = existing_records.toDF(["path", "content"]).withColumn(
#    "filename", split("path", "/").getItem(-1)).select("filename", "content")
#existing_records_df.cache()
#existing_records_df.show()
prompts_df.select("")
