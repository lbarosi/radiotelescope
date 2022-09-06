"""Form route."""
from flask import (
    Flask,
    url_for,
    render_template,
    redirect
)
from app.forms.forms import RunSDRForm

app = Flask(__name__, instance_relative_config=False)
app.config.from_object('config.Config')


@app.route("/", methods=["GET", "POST"])
def runSDR():
    """Run Standard form."""
    form = RunSDRForm()
    if form.validate_on_submit():
        return redirect(url_for("success"))
    return render_template(
        "runForm.jinja2",
        form=form,
        template="form-template"
    )
