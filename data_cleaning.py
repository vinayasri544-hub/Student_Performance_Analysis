import pandas as pd

DATA_FILE = "data/students.csv"


def clean_dataset():

    data = pd.read_csv(DATA_FILE)

    # Remove duplicates
    data = data.drop_duplicates()

        # Create average score
    data["average_score"] = data[
        [
            "Assignment Score",
            "Internal Mark",
            "Seminar Score",
            "Project Score",
            "Final Exam Marks"
        ]
    ].mean(axis=1)

        # Create average score
    data["average_score"] = data[
        [
            "Assignment Score",
            "Internal Mark",
            "Seminar Score",
            "Project Score",
            "Final Exam Marks"
        ]
    ].mean(axis=1)

        # Create grade
    def get_grade(score):
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    data["grade"] = data["average_score"].apply(get_grade)

        # Create attendance category
    def get_attendance_category(attendance):
        if attendance >= 85:
            return "High"
        elif attendance >= 75:
            return "Medium"
        else:
            return "Low"

    data["attendance_category"] = data["Attendance (%)"].apply(
        get_attendance_category
    )

    # Fill missing numeric values
    numeric_columns = data.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numeric_columns:
        data[column] = data[column].fillna(
            data[column].median()
        )

    # Fill missing text values
    text_columns = data.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:
        data[column] = data[column].fillna("Unknown")

    # Save cleaned dataset
    data.to_csv(
        "cleaned_student_data.csv",
        index=False
    )

    return data


if __name__ == "__main__":

    cleaned_data = clean_dataset()

    print("Data cleaning completed successfully.")
    print("Cleaned records:", len(cleaned_data))
    print("Cleaned columns:", len(cleaned_data.columns))