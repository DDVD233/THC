import pandas as pd


def main():
    df = pd.read_csv("CC200_ROI_labels.csv")
    aal_column = df["AAL"]
    new_aal_column = []
    for row in aal_column:
        # Get string between first and second quotation marks
        aal_number = row.split("\"")[1]
        new_aal_column.append(aal_number)
    df["AAL"] = new_aal_column
    df.to_csv("CC200_ROI_labels_clean.csv")
    print(df.head())


if __name__ == '__main__':
    main()