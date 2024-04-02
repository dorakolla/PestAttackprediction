from django import forms

# Existing city choices for Chhattisgarh
CITY_CHOICES_CHHATTISGARH = [
    ('raipur', 'Raipur'),
    ('bilaspur', 'Bilaspur'),
    ('bhilai', 'Bhilai'),
    ('durg', 'Durg'),
    ('korba', 'Korba'),
    ('raigarh', 'Raigarh'),
    ('jagdalpur', 'Jagdalpur'),
    ('ambikapur', 'Ambikapur'),
    ('rajnandgaon', 'Rajnandgaon'),
    ('dhamtari', 'Dhamtari'),
    ('mahasamund', 'Mahasamund'),
    ('kanker', 'Kanker'),
    ('janjgir', 'Janjgir'),
    ('kawardha', 'Kawardha'),
    ('bilaspur', 'Bilaspur'),  # Duplicate entry, remove if needed
    ('ambikapur', 'Ambikapur'),  # Duplicate entry, remove if needed
    ('rajnandgaon', 'Rajnandgaon'),  # Duplicate entry, remove if needed
    ('champa', 'Champa'),
    ('bhilai', 'Bhilai'),
    ('kumhari', 'Kumhari')
]

# Additional city choices for other states
STATE_CHOICES = [
    ('select a state','select a state'),
    ('Chhattisgarh', 'Chhattisgarh'),
    ('TamilNadu', 'TamilNadu'),
    ('WestBengal', 'WestBengal'),
    ('Karnataka', 'Karnataka'),
    ('Telangana', 'Telangana')
]

class CityForm(forms.Form):
    city = forms.ChoiceField(choices=CITY_CHOICES_CHHATTISGARH, widget=forms.Select(attrs={'class': 'form-control'}))
    state = forms.ChoiceField(choices=STATE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
