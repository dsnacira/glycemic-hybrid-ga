

## **README**

Le code développé sous Google Colab implémente un **cadre complet de simulation et d’optimisation pour le contrôle glycémique**, fondé sur une **architecture hybride auto-adaptative** combinant optimisation évolutionnaire, apprentissage supervisé et supervision par règles de sécurité. L’objectif principal est d’évaluer, de manière reproductible, l’impact de ces composantes sur la **stabilité glycémique**, la **sécurité** et le **coût computationnel**, dans des scénarios de vie quotidienne réalistes.

### **1\. Architecture générale du code**

L’implémentation est structurée de façon modulaire afin de séparer clairement :

* la **simulation physiologique**,  
* l’**optimisation génétique**,  
* l’**apprentissage du surrogate**,  
* les **règles de sécurité cliniques**,  
* et les **outils d’analyse et de visualisation**.

Cette modularité permet d’isoler chaque contribution méthodologique et de faciliter l’extension ou la reproduction des expériences.

### **2\. Simulation glycémique (Simulation Core)**

Le cœur du système repose sur un **simulateur discret de la glycémie**, exécuté sur un horizon de **24 heures** avec un pas temporel de **5 minutes** (288 échantillons par jour).

Le simulateur modélise :

* la dynamique du glucose sanguin,  
* l’**insulin-on-board (IOB)** via une décroissance exponentielle,  
* les effets des **repas**, du **stress** et de l’**activité physique**,  
* une **variabilité physiologique stochastique**,  
* et un terme de **régulation homéostatique** pour éviter les dérives longues.

À partir de la trajectoire glycémique simulée, les métriques cliniques standard sont calculées :

* **Time in Range (TIR)**,  
* **Time Below Range (TBR)**,  
* **Time Above Range (TAR)**,

puis combinées en une **fonction de fitness scalaire**, avec une pénalisation renforcée de l’hypoglycémie.

### **3\. Scénarios simulés**

Le code implémente **quatre scénarios standardisés**, utilisés de manière cohérente dans toutes les expériences :

1. **S1 – Journée normale** : repas équilibrés, pas de stress ni activité.  
2. **S2 – Jeûne / Ramadan** : absence de repas diurnes, repas unique en soirée.  
3. **S3 – Stress \+ repas riches en glucides** : augmentation glycémique induite par le stress.  
4. **S4 – Activité physique \+ faible apport glucidique** : augmentation de la sensibilité à l’insuline et risque hypoglycémique.

Ces scénarios sont **déterministes dans leur configuration**, ce qui garantit une comparaison équitable entre méthodes.

### **4\. Optimisation génétique (GA)**

L’optimisation repose sur un **algorithme génétique continu**, chargé d’optimiser un vecteur de décision à deux paramètres :

* le **débit basal d’insuline**,  
* la **durée d’action de l’insuline**.

Le GA implémente :

* sélection par tournoi,  
* croisement,  
* mutation,  
* élitisme.

Chaque génération produit une population de candidats, dont les performances sont évaluées soit par **simulation directe**, soit par **prédiction surrogate**, selon la méthode considérée.

### **5\. Surrogate learning (apprentissage supervisé)**

Afin de réduire le coût élevé des évaluations par simulation complète, le code intègre un **modèle surrogate supervisé** entraîné sur les données issues des simulations précédentes.

Le surrogate :

* apprend une approximation de la fitness réelle,  
* est mis à jour périodiquement,  
* permet de prédire la fitness d’une partie de la population,  
* n’est utilisé qu’après une **phase de warm-up** initiale.

Dans l’architecture hybride, **seule une fraction contrôlée** de la population est réellement simulée à chaque génération, garantissant un compromis entre **précision physiologique** et **efficacité computationnelle**.

### **6\. Supervision par règles de sécurité**

Un module de **règles cliniques explicites** est intégré pour empêcher l’exploration de régions dangereuses de l’espace de recherche. Ces règles incluent :

* des bornes sur les paramètres optimisés,  
* la détection d’hypoglycémie sévère (par ex. glycémie \< 54 mg/dL),  
* des règles dépendantes du scénario (jeûne, activité physique),  
* des mécanismes de **pénalisation ou de clampage**.

Cette supervision agit comme une couche de **sécurité interprétable**, complémentaire à l’optimisation et à l’apprentissage.

### **7\. Protocoles expérimentaux et reproductibilité**

Toutes les expériences sont répétées sur **plusieurs graines aléatoires**, et les résultats sont agrégés sous forme de moyennes et écarts-types.  
Le code génère automatiquement :

* les courbes de convergence,  
* les trajectoires glycémique optimales,  
* les erreurs de prédiction du surrogate,  
* et les statistiques cliniques finales.

Les paramètres expérimentaux sont centralisés dans des fichiers **YAML**, garantissant une **reproductibilité complète**.

### **8\. Rôle de Google Colab**

Google Colab a été utilisé comme environnement d’exécution pour :

* la simulation massive des scénarios,  
* l’entraînement du surrogate,  
* la génération des figures finales (PDF),  
* et l’export des résultats vers le manuscrit LaTeX.

Le code est entièrement compatible avec un environnement Python standard et peut être exécuté hors Colab sans modification conceptuelle.

### **9\. Finalité scientifique**

Ce code ne vise pas une application clinique directe, mais constitue un **outil de recherche reproductible** permettant :

* d’analyser les compromis entre efficacité computationnelle et robustesse glycémique,  
* d’évaluer l’intérêt des architectures hybrides dans des environnements ambiants,  
* et de fournir un cadre expérimental clair pour des comparaisons futures avec la littérature.


  
