from django.shortcuts import render
from .forms import CarForm
from PredictionModel.predict import evaluate


def index(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = CarForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            make = form.cleaned_data['make']
            model = form.cleaned_data['model']
            year = form.cleaned_data['year']
            odometer = form.cleaned_data['odometer']
            title = form.cleaned_data['title']
            condition = form.cleaned_data['condition']
            value = evaluate(make, model, year, odometer, title, condition)
            return render(request, 'evaluation/index.html', {'form': form, 'value': value})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = CarForm()

    return render(request, 'evaluation/index.html', {'form': form})
