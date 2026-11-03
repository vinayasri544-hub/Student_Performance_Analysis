from flask import Flask, render_template, jsonify, request, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
import re
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A3
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
from xml.sax.saxutils import escape


app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret"

# ==========================================================
# LOAD DATASET
# ==========================================================

DATA_FILE = "data/Student Performance Final.csv"

data = pd.read_csv(DATA_FILE)


# ==========================================================
# DATA CLEANING
# ==========================================================

original_records = len(data)

duplicate_records = int(data.duplicated().sum())

missing_values_before = int(
    data.isnull().sum().sum()
)

# Remove duplicates
cleaned_data = data.drop_duplicates().copy()


# Fill missing numeric values
numeric_columns = cleaned_data.select_dtypes(
    include=["number"]
).columns

for column in numeric_columns:
    cleaned_data[column] = cleaned_data[column].fillna(
        cleaned_data[column].median()
    )


# Fill missing text values
text_columns = cleaned_data.select_dtypes(
    include=["object"]
).columns

for column in text_columns:
    cleaned_data[column] = cleaned_data[column].fillna(
        "Unknown"
    )


# ==========================================================
# CREATE NEW COLUMNS
# ==========================================================

# Average Score
cleaned_data["average_score"] = cleaned_data[
    [
        "Assignment Score",
        "Internal Mark",
        "Seminar Score",
        "Project Score",
        "Final Exam Marks"
    ]
].mean(axis=1)


# Grade
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


cleaned_data["grade"] = cleaned_data[
    "average_score"
].apply(get_grade)


# Attendance Category
def get_attendance_category(attendance):

    if attendance >= 85:
        return "High"

    elif attendance >= 75:
        return "Medium"

    else:
        return "Low"


cleaned_data["attendance_category"] = cleaned_data[
    "Attendance (%)"
].apply(get_attendance_category)


# ==========================================================
# CLEANING INFORMATION
# ==========================================================

missing_values_after = int(
    cleaned_data.isnull().sum().sum()
)

cleaned_records = len(cleaned_data)

duplicates_removed = (
    original_records - cleaned_records
)


# ==========================================================
# BASIC INFORMATION
# ==========================================================

total_students = len(cleaned_data)

total_features = len(cleaned_data.columns)


# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

average_cgpa = round(
    cleaned_data["CGPA"].mean(),
    2
)

average_attendance = round(
    cleaned_data["Attendance (%)"].mean(),
    2
)

average_study_hours = round(
    cleaned_data["Study Hours/Day"].mean(),
    2
)

average_final_marks = round(
    cleaned_data["Final Exam Marks"].mean(),
    2
)


# ==========================================================
# PERFORMANCE LEVEL
# ==========================================================

performance_data = (
    cleaned_data["Performance Level"]
    .value_counts()
)

performance_labels = (
    performance_data.index.tolist()
)

performance_values = (
    performance_data.values.tolist()
)


# ==========================================================
# PLACEMENT STATUS
# ==========================================================

placement_data = (
    cleaned_data["Placement Status"]
    .value_counts()
)

placement_labels = (
    placement_data.index.tolist()
)

placement_values = (
    placement_data.values.tolist()
)


# ==========================================================
# AVERAGE MARKS
# ==========================================================

marks_columns = [
    "Assignment Score",
    "Internal Mark",
    "Seminar Score",
    "Project Score",
    "Final Exam Marks"
]

average_marks = []

for column in marks_columns:

    average_marks.append(
        round(
            cleaned_data[column].mean(),
            2
        )
    )
    
# ==========================================================
# GENDER-WISE PERFORMANCE
# ==========================================================

gender_performance = (
    cleaned_data
    .groupby("Gender")["Final Exam Marks"]
    .mean()
    .round(2)
)

gender_labels = gender_performance.index.tolist()

gender_values = gender_performance.values.tolist()


# ==========================================================
# TREND ANALYSIS
# ==========================================================

trend_data = (
    cleaned_data
    .groupby(["Year", "Semester"])
    .agg({
        "CGPA": "mean",
        "Final Exam Marks": "mean"
    })
    .reset_index()
)

trend_labels = (
    trend_data["Year"].astype(str)
    + " - Sem "
    + trend_data["Semester"].astype(str)
).tolist()

trend_cgpa = (
    trend_data["CGPA"]
    .round(2)
    .tolist()
)

trend_final_marks = (
    trend_data["Final Exam Marks"]
    .round(2)
    .tolist()
)


# ==========================================================
# PERFORMANCE PREDICTION - RANDOM FOREST
# ==========================================================

prediction_features = [
    "Attendance (%)",
    "Study Hours/Day",
    "Assignment Score",
    "Internal Mark",
    "Seminar Score",
    "Project Score",
    "Online Course Hours/Week",
    "Communication Score",
    "Aptitude Score",
    "Coding Score",
    "tech_test_score",
    "skill_rating",
    "projects_count",
    "certifications",
    "coding_hours_week",
    "problems_solved",
    "github_commits",
    "contest_rating",
    "aptitude_score",
    "mock_interview_score",
    "resume_score",
    "internships"
]

X = cleaned_data[
    prediction_features
]

y = cleaned_data[
    "Final Exam Marks"
]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

prediction_model = RandomForestRegressor(

    n_estimators=100,

    random_state=42
)

prediction_model.fit(
    X_train,
    y_train
)

y_pred = prediction_model.predict(
    X_test
)

prediction_mae = round(

    mean_absolute_error(
        y_test,
        y_pred
    ),

    2
)

prediction_r2 = round(

    r2_score(
        y_test,
        y_pred
    ),

    2
)


# ==========================================================
# AT-RISK STUDENT DETECTION
# ==========================================================

at_risk_data = cleaned_data[

    (cleaned_data["Attendance (%)"] < 75)

    |

    (cleaned_data["CGPA"] < 6.0)

    |

    (cleaned_data["Final Exam Marks"] < 50)

    |

    (cleaned_data["Arrears"] > 0)

].copy()

at_risk_count = len(
    at_risk_data
)

at_risk_table = at_risk_data[

    [
        "Student Reg No",
        "Student Name",
        "Department",
        "Year",
        "Semester",
        "Attendance (%)",
        "CGPA",
        "Final Exam Marks",
        "Arrears"
    ]

]


# ==========================================================
# STUDENT CLUSTERING - K-MEANS
# ==========================================================

clustering_features = [

    "Attendance (%)",
    "Study Hours/Day",
    "Final Exam Marks",
    "CGPA"

]

cluster_data = cleaned_data[
    clustering_features
].copy()

kmeans_model = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10

)

cluster_data["Cluster"] = (
    kmeans_model.fit_predict(
        cluster_data
    )
)

cluster_summary = (

    cluster_data

    .groupby("Cluster")[
        clustering_features
    ]

    .mean()

    .round(2)

)

cluster_counts = (

    cluster_data["Cluster"]

    .value_counts()

    .sort_index()

    .tolist()

)

cluster_labels = [

    "Cluster 1",
    "Cluster 2",
    "Cluster 3"

]


# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

correlation_columns = [

    "Attendance (%)",
        "Study Hours/Day",
        "Assignment Score",
        "Internal Mark",
        "Seminar Score",
        "Project Score",
        "Online Course Hours/Week",
        "Communication Score",
        "Aptitude Score",
        "Coding Score",
        "tech_test_score",
        "skill_rating",
        "projects_count",
        "certifications",
        "coding_hours_week",
        "problems_solved",
        "github_commits",
        "contest_rating",
        "aptitude_score",
        "mock_interview_score",
        "resume_score",
        "internships"

]

correlation_data = {}

for column in correlation_columns:

    correlation_data[column] = round(

        cleaned_data[column].corr(
            cleaned_data["CGPA"]
        ),

        2

    )


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/")
def dashboard():

    pass_percentage = round(
        (
            cleaned_data["Final Exam Marks"] >= 50
        ).mean() * 100,
        2
    )

    return render_template(

        "dashboard.html",

        total_students=total_students,

        total_features=total_features,

        missing_values=missing_values_before,

        duplicate_records=duplicate_records,

        average_marks=average_final_marks,

        pass_percentage=pass_percentage,

        at_risk_count=at_risk_count

    )


# ==========================================================
# ANALYSIS API
# ==========================================================

@app.route("/api/analysis")
def analysis_api():

    return jsonify({

        "performance_labels":
            performance_labels,

        "performance_values":
            performance_values,

        "placement_labels":
            placement_labels,

        "placement_values":
            placement_values,

        "marks_labels":
            marks_columns,

        "marks_values":
            average_marks,
                
        "gender_labels":
            gender_labels,

        "gender_values":
            gender_values

    })


# ==========================================================
# ANALYSIS PAGE
# ==========================================================

@app.route("/analysis")
def analysis():

    return render_template(

        "analysis.html",

        total_students=total_students,

        total_features=total_features,

        missing_values=missing_values_after,

        duplicate_records=duplicates_removed,

        average_cgpa=average_cgpa,

        average_attendance=average_attendance,

        average_study_hours=average_study_hours,

        average_final_marks=average_final_marks,

        performance_count=dict(
            zip(
                performance_labels,
                performance_values
            )
        ),

        placement_count=dict(
            zip(
                placement_labels,
                placement_values
            )
        ),

        performance_labels=performance_labels,

        performance_values=performance_values,

        placement_labels=placement_labels,

        placement_values=placement_values,

        marks_labels=marks_columns,

        marks_values=average_marks

    )


# ==========================================================
# TREND API
# ==========================================================

@app.route("/api/trends")
def trends_api():

    return jsonify({

        "labels":
            trend_labels,

        "cgpa":
            trend_cgpa,

        "final_marks":
            trend_final_marks

    })


# ==========================================================
# TREND PAGE
# ==========================================================

@app.route("/trends")
def trends():

    return render_template(

        "trends.html",

        trend_labels=trend_labels,

        trend_cgpa=trend_cgpa,

        trend_final_marks=trend_final_marks

    )


# ==========================================================
# CORRELATION API
# ==========================================================

@app.route("/api/correlation")
def correlation_api():

    correlation_columns = [
    "Attendance (%)",
    "Study Hours/Day",
    "Assignment Score",
    "Internal Mark",
    "Seminar Score",
    "Project Score",
    "Final Exam Marks",
    "CGPA",
    "Communication Score",
    "Aptitude Score",
    "Coding Score",
    "tech_test_score",
    "skill_rating",
    "projects_count",
    "certifications",
    "coding_hours_week",
    "problems_solved",
    "github_commits",
    "contest_rating",
    "aptitude_score",
    "mock_interview_score",
    "resume_score",
    "internships"
]

    correlation_matrix = (
        cleaned_data[correlation_columns]
        .corr()
        .round(2)
        .to_dict()
    )

    scatter_data = cleaned_data[
        ["Attendance (%)", "Final Exam Marks"]
    ].dropna()

    scatter_data = scatter_data.sample(
        min(500, len(scatter_data)),
        random_state=42
    )

    return jsonify({
        "correlation_matrix": correlation_matrix,
        "scatter": [
            {
                "attendance": row["Attendance (%)"],
                "marks": row["Final Exam Marks"]
            }
            for _, row in scatter_data.iterrows()
        ]
    })


# ==========================================================
# CORRELATION PAGE
# ==========================================================

@app.route("/correlation")
def correlation():

    return render_template(
        "correlation.html"
    )


# ==========================================================
# PREDICTION PAGE
# ==========================================================

@app.route("/prediction")
def prediction():

    return render_template(

        "prediction.html",

        prediction_mae=
            prediction_mae,

        prediction_r2=
            prediction_r2

    )


# ==========================================================
# AT-RISK PAGE
# ==========================================================

@app.route("/at-risk")
def at_risk():

    table = at_risk_table.to_html(

        index=False,

        classes="student-table"

    )

    return render_template(

        "at_risk.html",

        at_risk_count=
            at_risk_count,

        table=table

    )


# ==========================================================
# CLUSTERING PAGE
# ==========================================================

@app.route("/clustering")
def clustering():

    return render_template(

        "clustering.html",

        cluster_counts=
            cluster_counts

    )



# ==========================================================
# REPORT / OVERALL REVIEW PAGE
# ==========================================================


@app.route("/report")
def report():
    return render_template(
        "report.html",
        total_students=total_students,
        average_final_marks=average_final_marks,
        average_cgpa=average_cgpa,
        at_risk_count=at_risk_count,
        original_records=original_records,
        cleaned_records=cleaned_records,
        duplicates_removed=duplicates_removed,
        missing_values_after=missing_values_after,
        prediction_mae=prediction_mae,
        prediction_r2=prediction_r2
    )


# ==========================================================
# DATA CLEANING PAGE
# ==========================================================

@app.route("/cleaning")
def cleaning():

    return render_template(

        "cleaning.html",

        original_records=
            original_records,

        cleaned_records=
            cleaned_records,

        duplicate_records=
            duplicate_records,

        duplicates_removed=
            duplicates_removed,

        missing_values_before=
            missing_values_before,

        missing_values_after=
            missing_values_after,

        average_score=round(
            cleaned_data[
                "average_score"
            ].mean(),
            2
        ),

        grades=
            cleaned_data[
                "grade"
            ].value_counts().to_dict(),

        attendance_categories=
            cleaned_data[
                "attendance_category"
            ].value_counts().to_dict()

    )


# ==========================================================
# STUDENT DATA PAGE
# ==========================================================

@app.route("/students")
def students():

    table = cleaned_data.to_html(
        index=False,
        classes="student-table"
    )

    gender_options = sorted(
        cleaned_data["Gender"].dropna().unique().tolist()
    )

    grade_options = sorted(
        cleaned_data["grade"].dropna().unique().tolist()
    )

    attendance_options = sorted(
        cleaned_data["attendance_category"].dropna().unique().tolist()
    )

    return render_template(
        "students.html",
        table=table,
        gender_options=gender_options,
        grade_options=grade_options,
        attendance_options=attendance_options
    )

# TECHNICAL KNOWLEDGE FEATURE

@app.route("/technical")
def technical():
    df = cleaned_data.copy()

    df["technical_score"] = (
        df["tech_test_score"].clip(0, 100) * 0.40
        + (df["skill_rating"].clip(0, 5) / 5 * 100) * 0.25
        + (df["projects_count"].clip(0, 10) / 10 * 100) * 0.20
        + (df["certifications"].clip(0, 10) / 10 * 100) * 0.15
    ).round(2)

    students = df[
        [
            "Student Reg No",
            "Student Name",
            "Department",
            "technical_score"
        ]
    ].sort_values(
        "technical_score", ascending=False
    ).to_dict(orient="records")

    average_score = round(
        df["technical_score"].mean(), 2
    )

    return render_template(
        "technical.html",
        students=students,
        average_score=average_score
    )


# CODING SKILLS FEATURE

@app.route("/coding-skills")
def coding_skills():
    df = cleaned_data.copy()

    df["coding_score"] = (
        (df["coding_hours_week"].clip(0, 40) / 40 * 100) * 0.20
        + (df["problems_solved"].clip(0, 500) / 500 * 100) * 0.35
        + (df["github_commits"].clip(0, 500) / 500 * 100) * 0.25
        + (df["contest_rating"].clip(0, 2000) / 2000 * 100) * 0.20
    ).round(2)

    students = df[
        [
            "Student Reg No",
            "Student Name",
            "Department",
            "coding_score"
        ]
    ].sort_values(
        "coding_score", ascending=False
    ).to_dict(orient="records")

    average_score = round(df["coding_score"].mean(), 2)

    return render_template(
        "coding_skills.html",
        students=students,
        average_score=average_score
    )


# PLACEMENT PREPARATION FEATURE

@app.route("/placement-preparation")
def placement_preparation():
    df = cleaned_data.copy()

    df["placement_preparation_score"] = (
        (df["aptitude_score"].clip(0, 100)) * 0.20
        + (df["mock_interview_score"].clip(0, 100)) * 0.20
        + (df["resume_score"].clip(0, 100)) * 0.15
        + (df["Coding Score"].clip(0, 100)) * 0.15
        + (df["tech_test_score"].clip(0, 100)) * 0.15
        + (df["skill_rating"].clip(0, 5) / 5 * 100) * 0.05
        + (df["projects_count"].clip(0, 10) / 10 * 100) * 0.05
        + (df["internships"].clip(0, 5) / 5 * 100) * 0.05
    ).round(2)

    students = df[
        [
            "Student Reg No",
            "Student Name",
            "Department",
            "placement_preparation_score"
        ]
    ].sort_values(
        "placement_preparation_score",
        ascending=False
    ).to_dict(orient="records")

    average_score = round(
        df["placement_preparation_score"].mean(), 2
    )

    return render_template(
        "placement_preparation.html",
        students=students,
        average_score=average_score
    )


# PLACEMENT READINESS SCORE FEATURE

@app.route("/placement-readiness")
def placement_readiness():
    df = cleaned_data.copy()

    df["readiness_score"] = (
        df["aptitude_score"].clip(0, 100) * 0.20
        + df["mock_interview_score"].clip(0, 100) * 0.20
        + df["resume_score"].clip(0, 100) * 0.15
        + df["Coding Score"].clip(0, 100) * 0.15
        + df["tech_test_score"].clip(0, 100) * 0.15
        + (df["skill_rating"].clip(0, 5) / 5 * 100) * 0.05
        + (df["projects_count"].clip(0, 10) / 10 * 100) * 0.05
        + (df["internships"].clip(0, 5) / 5 * 100) * 0.05
    ).round(2)

    df["readiness_status"] = df["readiness_score"].apply(
        lambda score: "Ready" if score >= 70 else "Needs Improvement"
    )

    students = df[
        [
            "Student Reg No",
            "Student Name",
            "Department",
            "readiness_score",
            "readiness_status"
        ]
    ].sort_values(
        "readiness_score",
        ascending=False
    ).to_dict(orient="records")

    average_score = round(df["readiness_score"].mean(), 2)

    ready_count = int((df["readiness_score"] >= 70).sum())
    improvement_count = len(df) - ready_count

    return render_template(
        "placement_readiness.html",
        students=students,
        average_score=average_score,
        ready_count=ready_count,
        improvement_count=improvement_count
    )

# PLACEMENT CHANCE PREDICTOR

@app.route("/prediction")
def placement_prediction():
    print("Available data columns:", cleaned_data.columns.tolist())
    df = cleaned_data.copy()

    # Features used by the prediction model
    feature_columns = [
        "aptitude_score",
        "mock_interview_score",
        "resume_score",
        "Coding Score",
        "tech_test_score",
        "skill_rating",
        "projects_count",
        "internships"
    ]

    # Check whether all required columns exist
    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_columns:
        return (
            "Missing columns in dataset: "
            + ", ".join(missing_columns),
            400
        )

    # We need a real placement outcome column to train
    # a genuine placement classifier.
    target_column = "placement_status"

    if target_column not in df.columns:
        return (
            "Placement prediction needs a real placement outcome "
            "column named 'placement_status' in the dataset. "
            "Please check your dataset column name.",
            400
        )

    # Prepare data
    model_df = df[feature_columns + [target_column]].copy()

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna()

    # Convert the target into consistent text labels
    model_df[target_column] = (
        model_df[target_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep only known placement labels
    label_map = {
        "placed": 1,
        "yes": 1,
        "1": 1,
        "not placed": 0,
        "not_placed": 0,
        "no": 0,
        "0": 0
    }

    model_df[target_column] = model_df[target_column].map(label_map)
    model_df = model_df.dropna()

    if len(model_df) < 10 or model_df[target_column].nunique() < 2:
        return (
            "Not enough valid placement outcome records to train "
            "the predictor. Check placement_status values and dataset.",
            400
        )

    X = model_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    y = model_df[target_column].astype(int)

    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if len(X) < 10 or y.nunique() < 2:
        return "Not enough valid data to train the model.", 400

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Evaluate on held-out data
    y_pred = model.predict(X_test)
    model_accuracy = round(
        accuracy_score(y_test, y_pred) * 100, 2
    )

    # Predict placement probabilities for available students
    prediction_df = df.copy()
    prediction_X = prediction_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_students = prediction_X.notna().all(axis=1)
    prediction_df = prediction_df.loc[valid_students].copy()
    prediction_X = prediction_X.loc[valid_students]

    probabilities = model.predict_proba(prediction_X)
    placed_class_index = list(model.classes_).index(1)

    prediction_df["placement_chance"] = (
        probabilities[:, placed_class_index] * 100
    ).round(2)

    prediction_df["predicted_status"] = (
        prediction_df["placement_chance"]
        .apply(
            lambda chance:
                "Higher predicted chance"
                if chance >= 70
                else "Needs more preparation"
        )
    )

    # Rank the most influential input features
    importance = pd.DataFrame({
        "skill": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    feature_importance = importance.to_dict(orient="records")

    # Build student table
    student_columns = [
        col for col in
        ["Student Reg No", "Student Name", "Department"]
        if col in prediction_df.columns
    ]

    students = prediction_df[
        student_columns + ["placement_chance", "predicted_status"]
    ].sort_values(
        "placement_chance", ascending=False
    ).to_dict(orient="records")

    return render_template(
        "prediction.html",
        students=students,
        feature_importance=feature_importance,
        model_accuracy=model_accuracy,
        total_students=len(students)
    )


# ==========================================================
# FEATURE 7: MARKS VS SKILLS GAP
# ==========================================================

@app.route("/marks-skills-gap")
def marks_skills_gap():
    df = cleaned_data.copy()

    # Convert marks and coding scores to numeric values
    df["Final Exam Marks"] = pd.to_numeric(
        df["Final Exam Marks"], errors="coerce"
    )
    df["Coding Score"] = pd.to_numeric(
        df["Coding Score"], errors="coerce"
    )

    df = df.dropna(
        subset=["Final Exam Marks", "Coding Score"]
    )

    # Identify high academic marks but low coding scores
    gap_df = df[
        (df["Final Exam Marks"] >= 75)
        & (df["Coding Score"] < 50)
    ].copy()

    # Calculate the gap between marks and coding score
    gap_df["Marks Skills Gap"] = (
        gap_df["Final Exam Marks"] - gap_df["Coding Score"]
    ).round(2)

    gap_df = gap_df.sort_values(
        "Marks Skills Gap", ascending=False
    )

    columns = [
        "Student Reg No",
        "Student Name",
        "Department",
        "Final Exam Marks",
        "Coding Score",
        "Marks Skills Gap"
    ]

    students = gap_df[columns].to_dict(orient="records")

    return render_template(
        "marks_skills_gap.html",
        students=students,
        total_students=len(students)
    )


# FEATURE 6: WHAT-IF SIMULATOR

@app.route("/what-if", methods=["GET", "POST"])
def what_if_simulator():
    df = cleaned_data.copy()

    preferred_columns = [
        "aptitude_score",
        "mock_interview_score",
        "resume_score",
        "Coding Score",
        "tech_test_score",
        "skill_rating"
    ]

    score_columns = [
        col for col in preferred_columns
        if col in df.columns
    ]

    if not score_columns:
        return (
            "No supported score columns found. "
            "Please check the dataset column names.",
            400
        )

    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[score_columns] = df[score_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(subset=score_columns)

    if df.empty:
        return "No valid score data available.", 400

    student = df.iloc[0]
    features = []

    for i, col in enumerate(score_columns):
        features.append({
            "name": col,
            "key": f"score_{i}",
            "value": float(student[col]),
            "minimum": float(df[col].min()),
            "maximum": float(df[col].max())
        })

    def comparison_score(values):
        normalized = []

        for col in score_columns:
            minimum = float(df[col].min())
            maximum = float(df[col].max())
            value = float(values[col])

            if maximum == minimum:
                normalized.append(100.0)
            else:
                score = (
                    (value - minimum) /
                    (maximum - minimum)
                ) * 100

                normalized.append(
                    max(0.0, min(100.0, score))
                )

        return sum(normalized) / len(normalized)

    current_values = {
        col: float(student[col])
        for col in score_columns
    }

    current_score = comparison_score(current_values)

    # Result from the previous submission
    result = session.pop("what_if_result", None)

    if request.method == "POST":
        simulated_values = current_values.copy()

        for i, col in enumerate(score_columns):
            key = f"score_{i}"

            try:
                value = float(
                    request.form.get(key, student[col])
                )
            except (TypeError, ValueError):
                value = float(student[col])

            minimum = float(df[col].min())
            maximum = float(df[col].max())

            value = max(minimum, min(maximum, value))
            simulated_values[col] = value

        new_score = comparison_score(simulated_values)

        result = {
            "current_score": round(current_score, 2),
            "new_score": round(new_score, 2),
            "difference": round(new_score - current_score, 2)
        }

        # Save result and redirect to GET
        session["what_if_result"] = result
        return redirect(url_for("what_if_simulator"))

    return render_template(
        "what_if.html",
        features=features,
        result=result
    )

# ==========================================================
# STUDENT CAREER ADVISOR - FEATURE 1
# SKILL GAP FINDER
# ==========================================================

@app.route("/skill-gap")
def skill_gap():
    df = cleaned_data.copy()

    skill_columns = [
        "Coding Score",
        "Aptitude Score",
        "Communication Score"
    ]

    # Check that the required dataset columns exist
    missing_columns = [
        col for col in skill_columns
        if col not in df.columns
    ]

    if missing_columns:
        return (
            "Missing dataset columns: "
            + ", ".join(missing_columns),
            500
        )

    # Convert scores to numeric values
    for col in skill_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=skill_columns).copy()

    # Calculate overall skill average
    df["Skill Average"] = df[skill_columns].mean(axis=1).round(2)

    # Calculate the gap from the target score of 70
    for col in skill_columns:
        gap_col = col.replace(" Score", " Gap")
        df[gap_col] = (70 - df[col]).clip(lower=0).round(2)

    # Identify each student's weakest skill
    def find_weakest_skill(row):
        scores = {
            "Coding": row["Coding Score"],
            "Aptitude": row["Aptitude Score"],
            "Communication": row["Communication Score"]
        }
        weakest = min(scores, key=scores.get)

        if scores[weakest] >= 70:
            return "No major gap"

        return weakest

    df["Weakest Skill"] = df.apply(find_weakest_skill, axis=1)

    # Simple recommendation based on the weakest skill
    def recommend_action(row):
        if row["Weakest Skill"] == "Coding":
            return "Practice programming and solve coding problems"
        elif row["Weakest Skill"] == "Aptitude":
            return "Practice quantitative and logical reasoning"
        elif row["Weakest Skill"] == "Communication":
            return "Practice speaking, presentations and interviews"
        return "Maintain current skills and practise regularly"

    df["Recommended Action"] = df.apply(recommend_action, axis=1)

    # Sort students by lowest skill average first
    df = df.sort_values("Skill Average", ascending=True)

    columns_to_show = [
        "Student Reg No",
        "Student Name",
        "Department",
        "Coding Score",
        "Aptitude Score",
        "Communication Score",
        "Skill Average",
        "Weakest Skill",
        "Recommended Action"
    ]

    # Keep only columns that exist in the dataset
    columns_to_show = [
        col for col in columns_to_show if col in df.columns
    ]

    students = df[columns_to_show].to_dict(orient="records")

    return render_template(
        "skill_gap.html",
        students=students,
        total_students=len(students)
    )

# ==========================================================
# STUDENT CAREER ADVISOR - FEATURE 2
# ROLE RECOMMENDER
# ==========================================================


   
@app.route("/role-recommender")
def role_recommender():
    df = cleaned_data.copy()

    required_columns = [
        "Coding Score",
        "Aptitude Score",
        "Communication Score"
    ]

    for col in required_columns:
        if col not in df.columns:
            return f"Missing dataset column: {col}", 500

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns).copy()

    def recommend_role(row):
        coding = row["Coding Score"]
        aptitude = row["Aptitude Score"]
        communication = row["Communication Score"]

        scores = {
            "Data Analyst":
                aptitude * 0.45 + coding * 0.30 + communication * 0.25,

            "Web Developer":
                coding * 0.60 + communication * 0.15 + aptitude * 0.25,

            "Software Tester / QA Engineer":
                aptitude * 0.40 + communication * 0.30 + coding * 0.30,

            "Software Engineer":
                coding * 0.50 + aptitude * 0.30 + communication * 0.20,

            "Data Scientist":
                aptitude * 0.50 + coding * 0.40 + communication * 0.10,

            "Business Analyst":
                communication * 0.40 + aptitude * 0.40 + coding * 0.20
        }

        best_role = max(scores, key=scores.get)
        match_score = round(scores[best_role], 1)

        return pd.Series([best_role, match_score])

    df[["Recommended Role", "Role Match Score"]] = df.apply(
        recommend_role, axis=1
    )

    display_columns = [
        "Student Reg No",
        "Student Name",
        "Department",
        "Coding Score",
        "Aptitude Score",
        "Communication Score",
        "Recommended Role",
        "Role Match Score"
    ]

    display_columns = [
        col for col in display_columns if col in df.columns
    ]

    df = df.sort_values("Role Match Score", ascending=False)

    students = df[display_columns].to_dict(orient="records")

    return render_template(
        "role_recommender.html",
        students=students,
        total_students=len(students)
    )



# Company Fit & Placement Advisor

@app.route("/company-fit")
def company_fit():
    df = cleaned_data.copy()

    required_columns = [
        "Coding Score",
        "Aptitude Score",
        "Communication Score"
    ]

    for col in required_columns:
        if col not in df.columns:
            return f"Missing dataset column: {col}", 500
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns).copy()

    # Five company-type suitability scores
    df["Service Fit"] = (
        df["Coding Score"] * 0.30
        + df["Aptitude Score"] * 0.30
        + df["Communication Score"] * 0.40
    )

    df["Product Fit"] = (
        df["Coding Score"] * 0.45
        + df["Aptitude Score"] * 0.40
        + df["Communication Score"] * 0.15
    )

    df["Startup Fit"] = (
        df["Coding Score"] * 0.40
        + df["Aptitude Score"] * 0.25
        + df["Communication Score"] * 0.35
    )

    df["IT Consulting Fit"] = (
        df["Coding Score"] * 0.30
        + df["Aptitude Score"] * 0.35
        + df["Communication Score"] * 0.35
    )

    df["Testing QA Fit"] = (
        df["Coding Score"] * 0.25
        + df["Aptitude Score"] * 0.45
        + df["Communication Score"] * 0.30
    )

    fit_columns = [
        "Service Fit",
        "Product Fit",
        "Startup Fit",
        "IT Consulting Fit",
        "Testing QA Fit"
    ]

    df[fit_columns] = df[fit_columns].clip(0, 100).round(1)

    company_names = {
        "Service Fit": "Service-based companies",
        "Product Fit": "Product-based companies",
        "Startup Fit": "Startups",
        "IT Consulting Fit": "IT consulting companies",
        "Testing QA Fit": "Testing / QA companies"
    }

    def get_advice(row):
        scores = {
            company_names[col]: row[col]
            for col in fit_columns
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        top_recommendations = [
            {
                "company_type": name,
                "score": score
            }
            for name, score in ranked[:5]
        ]

        # Illustrative readiness threshold, not a real company cutoff
        eligible = [item for item in ranked if item[1] >= 60]

        weak_skills = []
        targets = {
            "Coding Score": 70,
            "Aptitude Score": 70,
            "Communication Score": 70
        }

        for skill, target in targets.items():
            current = row[skill]
            if current < target:
                weak_skills.append({
                    "skill": skill.replace(" Score", ""),
                    "current": round(current, 1),
                    "target": target,
                    "improvement": round(target - current, 1)
                })

        if not eligible:
            status = "Needs Improvement"
            improvement_plan = (
                "Build programming fundamentals, practise aptitude "
                "questions, improve communication, and complete a "
                "relevant project before applying."
            )
            future_targets = [
                "Service-based companies",
                "IT consulting companies",
                "Testing / QA companies"
            ]
        else:
            status = "Potential Match"
            improvement_plan = (
                "Continue practising interview questions and build "
                "projects related to the recommended company types."
            )
            future_targets = [item[0] for item in ranked[:3]]

        return pd.Series({
            "Top Recommendations": top_recommendations,
            "Placement Status": status,
            "Weak Skills": weak_skills,
            "Improvement Plan": improvement_plan,
            "Future Target Companies": future_targets
        })

    advice = df.apply(get_advice, axis=1)
    df = pd.concat([df, advice], axis=1)

    display_columns = [
        "Student Reg No",
        "Student Name",
        "Department",
        "Coding Score",
        "Aptitude Score",
        "Communication Score",
        "Service Fit",
        "Product Fit",
        "Startup Fit",
        "IT Consulting Fit",
        "Testing QA Fit",
        "Top Recommendations",
        "Placement Status",
        "Weak Skills",
        "Improvement Plan",
        "Future Target Companies"
    ]

    display_columns = [
        col for col in display_columns if col in df.columns
    ]

    students = df[display_columns].to_dict(orient="records")

    return render_template(
        "company_fit.html",
        students=students,
        total_students=len(students)
    )

# STUDENT CAREER ADVISOR - FEATURE 3


@app.route("/career-roadmap")
def career_roadmap():
    df = cleaned_data.copy()

    # Required columns
    required_columns = [
        "Student Reg No",
        "Student Name",
        "Coding Score",
        "Aptitude Score",
        "Communication Score"
    ]

    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            return f"Missing dataset column: {col}", 500

    # Department column may have a different name in some datasets
    if "Department" not in df.columns:
        possible_department_columns = [
            "department",
            "Department Name",
            "Dept",
            "Branch",
            "Course"
        ]

        found_column = next(
            (col for col in possible_department_columns if col in df.columns),
            None
        )

        if found_column:
            df["Department"] = df[found_column]
        else:
            df["Department"] = "Not Available"

    # Convert skill scores to numeric
    skill_columns = [
        "Coding Score",
        "Aptitude Score",
        "Communication Score"
    ]

    for col in skill_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=skill_columns).copy()

    def create_roadmap(row):
        skills = {
            "Coding": row["Coding Score"],
            "Aptitude": row["Aptitude Score"],
            "Communication": row["Communication Score"]
        }

        # Find weakest skill
        weakest_skill = min(skills, key=skills.get)
        current_skill_score = skills[weakest_skill]

        # Calculate role suitability scores
        role_scores = {
            "Data Analyst": (
                row["Aptitude Score"] * 0.45
                + row["Coding Score"] * 0.30
                + row["Communication Score"] * 0.25
            ),
            "Web Developer": (
                row["Coding Score"] * 0.60
                + row["Aptitude Score"] * 0.25
                + row["Communication Score"] * 0.15
            ),
            "Software Tester / QA Engineer": (
                row["Aptitude Score"] * 0.40
                + row["Communication Score"] * 0.30
                + row["Coding Score"] * 0.30
            ),
            "Software Engineer": (
                row["Coding Score"] * 0.50
                + row["Aptitude Score"] * 0.30
                + row["Communication Score"] * 0.20
            ),
            "Data Scientist": (
                row["Aptitude Score"] * 0.50
                + row["Coding Score"] * 0.40
                + row["Communication Score"] * 0.10
            ),
            "Business Analyst": (
                row["Communication Score"] * 0.40
                + row["Aptitude Score"] * 0.40
                + row["Coding Score"] * 0.20
            )
        }

        recommended_role = max(role_scores, key=role_scores.get)

        # Four-week plan based on weakest skill
        tasks = {
            "Coding": {
                "week1": "Revise programming fundamentals and syntax",
                "week2": "Solve 5 coding problems daily",
                "week3": "Build a small programming project",
                "week4": "Practise technical interviews"
            },
            "Aptitude": {
                "week1": "Revise percentages, ratios and averages",
                "week2": "Practise logical reasoning and timed tests",
                "week3": "Solve company-style aptitude mock tests",
                "week4": "Review mistakes and take a full mock test"
            },
            "Communication": {
                "week1": "Practise speaking for 10 minutes daily",
                "week2": "Prepare self-introduction and explain projects",
                "week3": "Practise group discussion and presentation",
                "week4": "Attend mock interviews and improve answers"
            }
        }

        plan = tasks[weakest_skill]

        return pd.Series({
            "Recommended Role": recommended_role,
            "Weakest Skill": weakest_skill,
            "Current Skill Score": current_skill_score,
            "Week 1 Task": plan["week1"],
            "Week 2 Task": plan["week2"],
            "Week 3 Task": plan["week3"],
            "Week 4 Task": plan["week4"],
            "Target Score": 70
        })

    # Generate roadmap for every student
    roadmap = df.apply(create_roadmap, axis=1)
    df = pd.concat([df, roadmap], axis=1)

    # Columns sent to the HTML template
    display_columns = [
        "Student Reg No",
        "Student Name",
        "Department",
        "Recommended Role",
        "Coding Score",
        "Aptitude Score",
        "Communication Score",
        "Weakest Skill",
        "Current Skill Score",
        "Week 1 Task",
        "Week 2 Task",
        "Week 3 Task",
        "Week 4 Task",
        "Target Score"
    ]

    students = df[display_columns].to_dict(orient="records")

    return render_template(
        "career_roadmap.html",
        students=students,
        total_students=len(students)
    )

# ==========================================================
# INDIVIDUAL STUDENT REPORT + SEARCH SUGGESTIONS + PDF
# ==========================================================

@app.route("/student-report", methods=["GET", "POST"])
def student_report():
    query = request.values.get("search", "").strip()

    student = None
    student_index = None

    # Get all student names for search suggestions
    student_names = (
        cleaned_data["Student Name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    student_names = sorted(
        [name for name in student_names.unique() if name],
        key=str.casefold
    )

    if query:
        df = cleaned_data.copy()

        # Search by student name
        name_match = df["Student Name"].astype(str).str.contains(
            query,
            case=False,
            na=False,
            regex=False
        )

        # Search by registration number
        reg_match = df["Student Reg No"].astype(str).str.contains(
            query,
            case=False,
            na=False,
            regex=False
        )

        matches = df[name_match | reg_match]

        if not matches.empty:
            student_index = int(matches.index[0])

            row = matches.iloc[0]

            student = {}

            for column, value in row.items():
                if pd.isna(value):
                    student[str(column)] = None
                elif isinstance(value, np.generic):
                    student[str(column)] = value.item()
                else:
                    student[str(column)] = value

    return render_template(
        "student_report.html",
        search=query,
        student=student,
        student_index=student_index,
        student_names=student_names
    )


# ==========================================================
# DOWNLOAD INDIVIDUAL STUDENT PDF
# ==========================================================

@app.route("/student-report/pdf/<int:student_index>")
def download_student_report(student_index):
    df = cleaned_data.copy()

    if student_index not in df.index:
        return "Student not found.", 404

    row = df.loc[student_index]

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A3),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "Individual Student Performance Report",
            styles["Title"]
        ),
        Spacer(1, 8 * mm)
    ]

    table_data = [["Field", "Student Details"]]

    for column, value in row.items():
        if pd.isna(value):
            display_value = "Not Available"
        else:
            display_value = str(value)

        table_data.append([
            Paragraph(
                escape(str(column)),
                styles["BodyText"]
            ),
            Paragraph(
                escape(display_value),
                styles["BodyText"]
            )
        ])

    table = Table(
        table_data,
        colWidths=[65 * mm, 300 * mm],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#243b64")
        ),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(table)
    pdf.build(story)

    buffer.seek(0)

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(row.get("Student Name", "student"))
    )

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{safe_name}_student_report.pdf",
        mimetype="application/pdf"
    )

# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
