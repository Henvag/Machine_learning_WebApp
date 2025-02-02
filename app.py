from flask import Flask, render_template, request, redirect, url_for, flash
from flask_babel import Babel, gettext as _
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Needed for flashing messages
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
babel = Babel(app)

def get_locale():
    return request.args.get('lang') or 'en'

babel.init_app(app, locale_selector=get_locale)

# Define available models for each dataset type
MODEL_OPTIONS = {
    'regression': [
        {'name': _('Linear Regression'), 'value': 'linear_regression'},
        {'name': _('Decision Tree Regressor'), 'value': 'decision_tree'},
        # Add more regression models here
    ],
    'classification': [
        {'name': _('Logistic Regression'), 'value': 'logistic_regression'},
        {'name': _('Decision Tree Classifier'), 'value': 'decision_tree_classifier'},
        # Add more classification models here
    ]
}

@app.context_processor
def inject_functions():
    return dict(get_locale=get_locale)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    chosen_dataset = None
    chosen_model_type = None
    chosen_model_name = None
    train_score = None
    test_score = None
    metric_name_train = None
    metric_name_test = None
    available_models = MODEL_OPTIONS['regression']  # Default models

    if request.method == "POST":
        chosen_dataset = request.form.get("dataset")
        chosen_model_type = request.form.get("model_type")

        # Determine dataset type and load data
        if chosen_dataset == "diabetes":
            dataset_type = 'regression'
            X, y = load_diabetes(return_X_y=True)
            metric_name_train = _("R2 Score")
            metric_name_test = _("R2 Score")
        elif chosen_dataset == "iris":
            dataset_type = 'classification'
            X, y = load_iris(return_X_y=True)
            metric_name_train = _("Accuracy")
            metric_name_test = _("Accuracy")
        else:
            flash(_("Invalid dataset selected."), "danger")
            return redirect(url_for('index'))

        # Update available models based on dataset type
        available_models = MODEL_OPTIONS.get(dataset_type, [])

        # Validate model type
        valid_model_types = [model['value'] for model in MODEL_OPTIONS[dataset_type]]
        if chosen_model_type not in valid_model_types:
            flash(_("Invalid model type selected."), "danger")
            return redirect(url_for('index'))

        # Retrieve the selected model's display name
        selected_model = next((model for model in MODEL_OPTIONS[dataset_type] if model['value'] == chosen_model_type), None)
        if selected_model:
            chosen_model_name = selected_model['name']
        else:
            flash(_("Selected model type is not implemented."), "danger")
            return redirect(url_for('index'))

        # Initialize the selected model
        if chosen_model_type == "linear_regression":
            model = LinearRegression()
            metric_func = r2_score
        elif chosen_model_type == "decision_tree":
            model = DecisionTreeRegressor()
            metric_func = r2_score
        elif chosen_model_type == "logistic_regression":
            model = LogisticRegression(max_iter=200)
            metric_func = accuracy_score
        elif chosen_model_type == "decision_tree_classifier":
            model = DecisionTreeClassifier()
            metric_func = accuracy_score
        else:
            flash(_("Selected model type is not implemented."), "danger")
            return redirect(url_for('index'))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Format scores
        train_score = f"{metric_func(y_train, y_train_pred):.2f}"
        test_score = f"{metric_func(y_test, y_test_pred):.2f}"

        # Generate plot
        plt.figure(figsize=(8, 6))
        if dataset_type == 'regression':
            plt.scatter(y_test, y_test_pred, color='blue', edgecolors='k', alpha=0.7)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
            plt.xlabel(_('Actual Values'))
            plt.ylabel(_('Predicted Values'))
            plt.title(_('Actual vs Predicted Values'))
        elif dataset_type == 'classification':
            cm = confusion_matrix(y_test, y_test_pred)
            # For iris dataset, load target names; otherwise, set to None
            if chosen_dataset == "iris":
                target_names = load_iris().target_names
            else:
                target_names = None
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(_('Confusion Matrix'))

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plot_url = base64_image
        plt.close()

        flash(_("Model trained successfully!"), "success")

    return render_template("index.html",
                           chosen_dataset=chosen_dataset,
                           chosen_model_type=chosen_model_type,
                           chosen_model_name=chosen_model_name,
                           train_score=train_score,
                           test_score=test_score,
                           model_options=MODEL_OPTIONS,
                           metric_name_train=metric_name_train,
                           metric_name_test=metric_name_test,
                           plot_url=plot_url)  # Added plot_url to context

@app.route("/change_language/<language>")
def change_language(language):
    return redirect(url_for('index', lang=language))

if __name__ == "__main__":
    app.run(debug=True)