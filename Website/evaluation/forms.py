from django import forms


class CarForm(forms.Form):
    make = forms.CharField()
    model = forms.CharField()
    year = forms.IntegerField()
    odometer = forms.IntegerField(min_value=0)
    title = forms.CharField(required=False)
    condition = forms.CharField(required=False)
