import torch
import os

def train_engine(model, train_loader, val_loader, criterion, optimizer, device, patience=5, epochs=100):
    best_val_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        t_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()  # Backpropagation
            optimizer.step() # Weight update
            t_loss += loss.item()
        
        # --- VALIDATION PHASE ---
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                v_out = model(X)
                v_loss += criterion(v_out, y).item()

        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss {avg_t:.4f} | Val Loss {avg_v:.4f}")

        # --- EARLY STOPPING LOGIC ---
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            # Save the best model state
            if not os.path.exists('models'): os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                break
                
    return history