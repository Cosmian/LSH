import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm


D = 512  # Nombre de vecteurs aléatoires dans chaque famille
L = 50   # Nombre de tables de hachage
alpha = 0.1  # Facteur de réduction pour le filtrage
iProb = 3  # Nombre de meilleures projections à considérer pour chaque vecteur
k = 20  # Nombre de voisins les plus proches
qProbe = 10  # Nombre de buckets voisins sondés

# Fonction de création des vecteurs de hachage aléatoires
def creer_vecteurs_hachage(dimension, D):
    return np.random.randn(D, dimension)

# Calcul du hachage pour le vecteur
def calcul_hachage(vecteur, vecteurs_hachage1, vecteurs_hachage2, iProb):
    projections1 = vecteur @ vecteurs_hachage1.T
    meilleurs_indices1 = np.argsort(np.abs(projections1))[-iProb:]
    
    projections2 = vecteur @ vecteurs_hachage2.T
    meilleurs_indices2 = np.argsort(np.abs(projections2))[-iProb:]
    
    hachages = []
    for indice1 in meilleurs_indices1:
        signe1 = int(np.sign(projections1[indice1]))
        for indice2 in meilleurs_indices2:
            signe2 = int(np.sign(projections2[indice2]))
            hachages.append((indice1, signe1, indice2, signe2))
    return hachages


# Creation de l'indexe
def indexation(donnees, D, L, dimension, alpha, iProb):
    index = [{} for _ in range(L)]
    vecteurs_hachage1 = [creer_vecteurs_hachage(dimension, D) for _ in range(L)]
    vecteurs_hachage2 = [creer_vecteurs_hachage(dimension, D) for _ in range(L)]

    for vecteur in tqdm(donnees, desc="Création de l'index"):
        for i in range(L):
            hachages = calcul_hachage(vecteur, vecteurs_hachage1[i], vecteurs_hachage2[i], iProb)
            for hachage in hachages:
                if hachage not in index[i]:
                    index[i][hachage] = []
                index[i][hachage].append(vecteur)
                
    return index, vecteurs_hachage1, vecteurs_hachage2



# Fonction de filtrage des buckets
def filtrer_buckets(index, vecteurs_hachage1, vecteurs_hachage2, k, alpha, iProb):
    for i, table in enumerate(index):
        for hachage, bucket in table.items():
            B = len(bucket)
            if B > k:
                taille_cible = int((B * alpha) / iProb) +1
                vecteur_ref = vecteurs_hachage1[i][hachage[0]] * hachage[1] + vecteurs_hachage2[i][hachage[2]] * hachage[3]
                bucket.sort(key=lambda v: np.linalg.norm(v - vecteur_ref))
                table[hachage] = bucket[:taille_cible]

# Fonction de recherche pour une liste de requêtes
def recherche(queries, index, vecteurs_hachage1, vecteurs_hachage2, k, qProbe):
    resultats_approx = []
    for vecteur_query in tqdm(queries,desc='recherche'):
        candidats = set()
        
        for i in range(L):
            # Calcul des `qProbe` meilleures projections
            projections1 = vecteur_query @ vecteurs_hachage1[i].T
            meilleurs_indices1 = np.argsort(np.abs(projections1))[-qProbe:]
            
            projections2 = vecteur_query @ vecteurs_hachage2[i].T
            meilleurs_indices2 = np.argsort(np.abs(projections2))[-qProbe:]
            
            # Ajouter les vecteurs des buckets voisins dans `candidats`
            for indice1 in meilleurs_indices1:
                signe1 = int(np.sign(projections1[indice1]))
                for indice2 in meilleurs_indices2:
                    signe2 = int(np.sign(projections2[indice2]))
                    hachage = (indice1, signe1, indice2, signe2)
                    if hachage in index[i]:
                        candidats.update(map(tuple, index[i][hachage]))
        
        # Recherche brute pour trouver les k plus proches voisins dans les candidats
        candidats = [np.array(c) for c in candidats]
        distances = [np.linalg.norm(v - vecteur_query) for v in candidats]
        k_plus_proches_indices = np.argsort(distances)[:k]
        k_plus_proches = [candidats[idx] for idx in k_plus_proches_indices]
        
        resultats_approx.append(k_plus_proches)
    return resultats_approx

# Fonction pour calculer le taux de rappel (recall rate) pour une liste de requêtes
def calcul_recall_rate(queries, donnees, resultats_approx, k):
    recall_rates = []
    for vecteur_query, resultat_approx in zip(queries, resultats_approx):
        # Recherche brute sur tout l'ensemble de données pour les k plus proches voisins
        distances_reelles = [np.linalg.norm(v - vecteur_query) for v in donnees]
        indices_reels = np.argsort(distances_reelles)[:k]
        vrais_k_plus_proches = [donnees[idx] for idx in indices_reels]
        
        # Calcul du recall rate pour cette requête
        set_resultat_approx = set(map(tuple, resultat_approx))
        set_vrais_k_plus_proches = set(map(tuple, vrais_k_plus_proches))
        
        intersection = set_resultat_approx.intersection(set_vrais_k_plus_proches)
        recall_rate = len(intersection) / k
        recall_rates.append(recall_rate)

    return np.mean(recall_rates)


dimension = 384  # Dimension des vecteurs
donnees = np.load('maaarco.npy')
queries = np.load('maaarco_queries.npy')

donnees = donnees[:5000]
queries = queries[:100]


index, vecteurs_hachage1, vecteurs_hachage2 = indexation(donnees, D, L, dimension, alpha, iProb)


filtrer_buckets(index, vecteurs_hachage1, vecteurs_hachage2, k, alpha, iProb)

resultats_approx = recherche(queries, index, vecteurs_hachage1, vecteurs_hachage2, k, qProbe)

recall_rate_moyen = calcul_recall_rate(queries, donnees, resultats_approx, k)

print("recall rate :", recall_rate_moyen)
