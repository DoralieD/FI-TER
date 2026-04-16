import torch

print(f"Version de PyTorch : {torch.__version__}")

# Voici la bonne syntaxe pour la version de CUDA :
print(f"Version de CUDA : {torch.version.cuda}")

# On vérifie que la carte graphique est bien détectée avant de l'interroger
if torch.cuda.is_available():
    print("Architectures supportées :", torch.cuda.get_arch_list())
    print("Propriétés du GPU :", torch.cuda.get_device_properties(0))
    
    # Test : on envoie un tenseur (calcul) directement sur ta RTX 5060
    print("Test tenseur sur le GPU :", torch.randn(1).cuda())
else:
    print("Aïe, CUDA n'est pas disponible. PyTorch utilise le processeur au lieu de ta carte graphique !")