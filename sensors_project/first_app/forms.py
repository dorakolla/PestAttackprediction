from django import forms
CITY_CHOICES = [
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
('bilaspur', 'Bilaspur'),
('ambikapur', 'Ambikapur'),
('rajnandgaon', 'Rajnandgaon'),
('champa', 'Champa'),
('bhilai', 'Bhilai'),
('kumhari', 'Kumhari')
]

# Now, the "districts_in_chhattisgarh" list contains the districts of Chhattisgarh in the desired format.

class CityForm(forms.Form):
    city = forms.ChoiceField(choices=CITY_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
