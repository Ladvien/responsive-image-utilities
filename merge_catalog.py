import pandas as pd

ORIGINAL_DATASET = "/home/ladvien/laion-aesthetics-12m-umap-urls-and-hashes.csv"
DOWNLOADED_CATALOG = "/home/ladvien/catalog.csv"

df_original = pd.read_csv(ORIGINAL_DATASET)
df_downloaded = pd.read_csv(DOWNLOADED_CATALOG)

# print(df_original.head())
# print(df_downloaded.head())\

"""
Remove any rows where the filename contains two periods, e.g, `filename..jpg'
"""

df_downloaded = df_downloaded[
    ~df_downloaded["filename"].str.contains(r"\.\.")
].reset_index(drop=True)


"""
In the df_downloaded, split the field `filename' into `hash' and `extension', 
and convert it to int
"""

df_downloaded["hash"] = df_downloaded["filename"].apply(lambda x: x.split(".")[0])
df_downloaded["hash"] = df_downloaded["hash"].apply(lambda x: int(x))

print(df_original.head())
print(df_downloaded.head())

merged = pd.merge(df_original, df_downloaded, on="hash", how="left")
print(merged.head())


"""
Count the number of rows where the `filename' is NaN and
the number where thew rows have a value
"""

print(merged["filename"].isna().sum())
print(merged["filename"].notna().sum())

merged.to_csv("/home/ladvien/merged.csv", index=False)
