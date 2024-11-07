import numpy as np
from tqdm import tqdm


D = 384  # Nombre de vecteurs aléatoires dans chaque famille
L = 5   # Nombre de tables de hachage
alpha = 0.1  # Facteur de réduction pour le filtrage
iProb = 3  # Nombre de meilleures projections à considérer pour chaque vecteur
k = 10  # Nombre de voisins les plus proches
qProbe = 3  # Nombre de buckets voisins sondés


def creer_vecteurs_hachage(dimension, D):
    return np.random.randn(D, dimension)


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


def indexation(donnees, D, L, dimension, alpha, iProb):
    index = [{} for _ in range(L)]
    vecteurs_hachage1 = normalise([creer_vecteurs_hachage(dimension, D) for _ in range(L)])
    vecteurs_hachage2 = normalise([creer_vecteurs_hachage(dimension, D) for _ in range(L)])

    for vecteur in tqdm(donnees, desc="Création de l'index"):
        for i in range(L):
            hachages = calcul_hachage(vecteur, vecteurs_hachage1[i], vecteurs_hachage2[i], iProb)
            for hachage in hachages:
                vecteur_ref1 = vecteurs_hachage1[i][hachage[0]] * hachage[1]
                vecteur_ref2 = vecteurs_hachage2[i][hachage[2]] * hachage[3]
                score = min(np.dot(vecteur, vecteur_ref1), np.dot(vecteur, vecteur_ref2))
                if hachage not in index[i]:
                    index[i][hachage] = []
                index[i][hachage].append((vecteur, score))
                
    return index, vecteurs_hachage1, vecteurs_hachage2

# Fonction de filtrage des buckets en utilisant les scores précalculés
def filtrer_buckets(index, k, alpha, iProb):
    for table in index:
        for hachage, bucket in table.items():
            B = len(bucket)
            if B > k:
                taille_cible = int((B * alpha) / iProb)
                # Trier le bucket selon le score en ordre décroissant
                bucket.sort(key=lambda x: -x[1])  # x[1] est le score
                table[hachage] = bucket[:taille_cible]

def recherche(queries, index, vecteurs_hachage1, vecteurs_hachage2, k, qProbe):
    resultats_approx = []
    list=[]
    for vecteur_query in tqdm(queries, desc='Recherche'):
        candidats = []
        cpt=0
        for i in range(L):
            projections1 = vecteur_query @ vecteurs_hachage1[i].T
            projections2 = vecteur_query @ vecteurs_hachage2[i].T
            for indice1 in np.argsort(np.abs(projections1))[-qProbe:]:
                signe1 = int(np.sign(projections1[indice1]))
                for indice2 in np.argsort(np.abs(projections2))[-qProbe:]:
                    signe2 = int(np.sign(projections2[indice2]))
                    hachage = (indice1, signe1, indice2, signe2)
                    if hachage in index[i]:
                        # On ajoute seulement les vecteurs, en ignorant les scores :
                        bucket = [vecteur for vecteur, score in index[i][hachage]]
                        candidats.extend(bucket)
                        cpt+=len(bucket)
        list.append(cpt)
        if candidats:
            k_plus_proches = sorted(candidats, key=lambda v: -np.dot(v,vecteur_query))[:k]
            resultats_approx.append(k_plus_proches)
        else:
            resultats_approx.append([]) 
    return resultats_approx,list



def calcul_recall_rate(queries, donnees, resultats_approx, k):
    recall_rates = []

    for vecteur_query, resultat_approx in zip(tqdm(queries,desc='brute'), resultats_approx):
        # Recherche brute sur tout l'ensemble de données 
        distances_reelles = [-np.dot(v , vecteur_query) for v in donnees]
        indices_reels = np.argsort(distances_reelles)[:k]
        vrais_k_plus_proches = [donnees[idx] for idx in indices_reels]
        
        # Calcul du recall rate 
        set_resultat_approx = set(map(tuple, resultat_approx))
        set_vrais_k_plus_proches = set(map(tuple, vrais_k_plus_proches))

        intersection = set_resultat_approx.intersection(set_vrais_k_plus_proches)
        recall_rate = len(intersection) / k
        recall_rates.append(recall_rate)

    return np.mean(recall_rates)

def normalise(vecteurs):
    vecteurs_normalises = []
    for vecteur in vecteurs:
        norme = np.linalg.norm(vecteur)
        vecteur_normalise = vecteur / norme  
        vecteurs_normalises.append(vecteur_normalise)
    return vecteurs_normalises

# Chargement des données et des requêtes
dimension = 384  # Dimension des vecteurs
donnees = np.load('maaarco.npy')
queries = np.load('maaarco_queries.npy')

#donnees = donnees[:50000]
#queries = queries[:100]

donnees = normalise(donnees)
queries = normalise(queries)



# Création de l'index et filtrage des buckets
index, vecteurs_hachage1, vecteurs_hachage2 = indexation(donnees, D, L, dimension, alpha, iProb)
filtrer_buckets(index, k, alpha, iProb)

resultats_approx = recherche(queries, index, vecteurs_hachage1, vecteurs_hachage2, k, qProbe)
print(' nombre de vecteurs renoyés par requete : ',np.mean(resultats_approx[1]))

recall_rate_moyen = calcul_recall_rate(queries, donnees, resultats_approx[0], k)

print("recall rate :", recall_rate_moyen)

print()
print('D=',D,'L=',L)
