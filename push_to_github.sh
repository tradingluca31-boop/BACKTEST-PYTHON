#!/bin/bash
# Script pour pousser le backup sur GitHub
# Usage: ./push_to_github.sh

echo "🚀 Push du backup vers GitHub..."
echo ""

# Vérifier si on est dans le bon répertoire
if [ ! -f "app_2025-10-01_04-18_backup.py" ]; then
    echo "❌ Erreur: fichier backup non trouvé"
    echo "Exécutez ce script depuis /tmp/BACKTEST-PYTHON/"
    exit 1
fi

# Afficher le statut
echo "📊 Statut Git:"
git status

echo ""
echo "📤 Push vers GitHub..."

# Push avec authentification interactive
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Backup sauvegardé avec succès sur GitHub!"
    echo "🔗 Voir: https://github.com/tradingluca31-boop/BACKTEST-PYTHON/blob/main/app_2025-10-01_04-18_backup.py"
else
    echo ""
    echo "❌ Erreur lors du push"
    echo "💡 Vous devrez peut-être vous authentifier avec GitHub CLI ou un token"
fi
