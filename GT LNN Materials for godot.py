import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'material': ['cloth', 'wood', 'metal', 'plastic'],
    'texture': ['soft', 'rough', 'smooth', 'smooth'],
    'color': ['varied', 'brown', 'grey', 'varied'],
    'elasticity': [0.8, 0.3, 0.1, 0.5],
    'density': [0.5, 0.7, 7.8, 0.9]
}

df = pd.DataFrame(data)

# Preprocess the data
encoder = OneHotEncoder()
encoded_material = encoder.fit_transform(df[['material']]).toarray()

scaler = StandardScaler()
scaled_properties = scaler.fit_transform(df[['elasticity', 'density']])

X = pd.concat([pd.DataFrame(encoded_material), pd.DataFrame(scaled_properties)], axis=1)
y = df[['texture', 'color']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import torch
import torch.nn as nn
import torch.optim as optim

class MaterialTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward):
        super(MaterialTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        transformer_out = self.transformer(src)
        out = self.fc(transformer_out)
        return out

# Parameters
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
nhead = 2
num_encoder_layers = 2
dim_feedforward = 512

model = MaterialTransformer(input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    X_batch = torch.tensor(X_train.values, dtype=torch.float32)
    y_batch = torch.tensor(y_train.values, dtype=torch.float32)
    
    output = model(X_batch)
    loss = criterion(output, y_batch)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
