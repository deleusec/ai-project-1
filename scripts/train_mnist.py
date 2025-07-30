import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Couches de convolution
        self.conv_layers = nn.Sequential(
            # Première couche de convolution
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Dropout(0.25),
            
            # Deuxième couche de convolution
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Dropout(0.25),
            
            # Troisième couche de convolution
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        
        # Couches fully connected
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),  # 7x7x128 = 6272
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),  # 10 classes pour MNIST
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

def get_data():
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=5): 
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.5)
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), '../models/mnist.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}% (Best: {best_accuracy:.2f}%)')
        scheduler.step()

def export_to_onnx(model):
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        "../public/models/mnist.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    print("Modèle exporté vers public/models/mnist.onnx")

if __name__ == "__main__":
    # Création du réseau de neurones
    model = NeuralNetwork()
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Chargement des données MNIST
    train_loader, test_loader = get_data()
    print("Données MNIST chargées")

    # Entraînement du modèle
    train_model(model, train_loader, test_loader, epochs=5)

    # Chargement du modèle
    model.load_state_dict(torch.load('../models/mnist.pth'))
    print("Meilleur modèle chargé pour l'exportation")

    # Exportation du modèle
    torch.save(model.state_dict(), '../models/mnist.pth')
    print("Modèle sauvegardé: models/mnist.pth")
    
    # Exportation du modèle
    export_to_onnx(model)
    print("Entraînement terminé!")




