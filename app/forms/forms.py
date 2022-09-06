"""Form object declaration."""
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, DecimalField, IntegerField, SelectField
from wtforms import validators

class RunSDRForm(FlaskForm):
    """Parâmetros de entrada."""
    name = StringField("Name", [validators.InputRequired()])
    gain = DecimalField("Gain", [validators.NumberRange(min=0, max=50), validators.InputRequired()])
    fs = DecimalField("Sampling Frequency (Hz)", [validators.NumberRange(min=50000, max=2400000), validators.InputRequired()])
    fc = DecimalField("Central Frequency (Hz)", [validators.NumberRange(min=50000, max=1700000000), validators.InputRequired()])
    duration = DecimalField("Central Frequency (s)", [validators.NumberRange(min=0, max=900), validators.InputRequired()])
    n_channels = IntegerField("Number of FFT channels", [validators.NumberRange(min=1, max=500000), validators.InputRequired()])
    n_int = IntegerField("Number os samples to integrate", [validators.NumberRange(min=1, max=900), validators.InputRequired()])
    submit = SubmitField("Submit")
