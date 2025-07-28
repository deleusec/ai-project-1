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
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits   

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
    test_dataset = datasets.MNIST('data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=15): 
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
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
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
    train_model(model, train_loader, test_loader, epochs=15)

    # Chargement du modèle
    model.load_state_dict(torch.load('../models/mnist.pth'))
    print("Meilleur modèle chargé pour l'exportation")

    # Exportation du modèle
    torch.save(model.state_dict(), '../models/mnist.pth')
    print("Modèle sauvegardé: models/mnist.pth")
    
    # Exportation du modèle
    export_to_onnx(model)
    print("Entraînement terminé!")




