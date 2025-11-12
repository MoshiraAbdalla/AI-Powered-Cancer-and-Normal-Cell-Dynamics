from scipy.io import loadmat

# Load .mat file
data = loadmat('Sample Data Velocity Fields\VelocityField.mat')

# Check contents
print(data.keys())

# Access a variable
x = data['variable_name']
print(x)
