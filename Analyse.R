# =============================================================================
#  📱 MTN BÉNIN — Customer Churn Segmentation & Revenue Optimization
#  Language : R
#  Méthodes : K-Means, PCA, Logistic Regression, Random Forest, XGBoost
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 0. INSTALLATION & CHARGEMENT DES PACKAGES
# ─────────────────────────────────────────────────────────────────────────────

required_packages <- c(
  "tidyverse",   # manipulation et visualisation des données
  "caret",       # framework ML (train/test split, cross-validation)
  "xgboost",     # modèle XGBoost
  "randomForest",# modèle Random Forest
  "cluster",     # clustering (silhouette)
  "factoextra",  # visualisation clustering et PCA
  "ggplot2",     # graphiques
  "pROC",        # courbes ROC / AUC
  "scales",      # formatage des axes
  "gridExtra",   # mise en page des graphiques
  "corrplot"     # matrice de corrélation
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

set.seed(42)   # reproductibilité globale

# ─────────────────────────────────────────────────────────────────────────────
# 1. GÉNÉRATION DES DONNÉES SIMULÉES
#    (Dataset inspiré de la base abonnés MTN Bénin)
# ─────────────────────────────────────────────────────────────────────────────

n <- 2000   # nombre de clients simulés

# --- Simulation des variables selon 3 profils latents ---
# Cluster 1 : Haut risque  (~30 %)
# Cluster 2 : Risque moyen (~40 %)
# Cluster 3 : Faible risque (~30 %)

profil <- sample(c("high","medium","low"), n,
                 replace = TRUE, prob = c(0.30, 0.40, 0.30))

simule_clients <- function(profil) {
  dplyr::case_when(
    # tenure_months
    profil == "high"   ~ list(tenure = round(runif(1, 1, 12)),
                              charge  = round(rnorm(1, 8000, 1500)),
                              data    = round(runif(1, 0.2, 2.0), 2),
                              calls   = round(rnorm(1, 80, 30)),
                              compl   = sample(3:8, 1),
                              delay   = sample(2:6, 1),
                              churn   = rbinom(1, 1, 0.75)),
    profil == "medium" ~ list(tenure = round(runif(1, 12, 36)),
                              charge  = round(rnorm(1, 14000, 3000)),
                              data    = round(runif(1, 2.0, 6.0), 2),
                              calls   = round(rnorm(1, 200, 60)),
                              compl   = sample(1:3, 1),
                              delay   = sample(0:3, 1),
                              churn   = rbinom(1, 1, 0.35)),
    TRUE               ~ list(tenure = round(runif(1, 36, 72)),
                              charge  = round(rnorm(1, 22000, 4000)),
                              data    = round(runif(1, 6.0, 20.0), 2),
                              calls   = round(rnorm(1, 380, 80)),
                              compl   = sample(0:1, 1),
                              delay   = sample(0:1, 1),
                              churn   = rbinom(1, 1, 0.08))
  )
}

# Construction du data.frame ligne par ligne
rows <- lapply(seq_len(n), function(i) {
  p <- profil[i]
  if (p == "high") {
    data.frame(
      customer_id    = sprintf("MTN-%04d", i),
      tenure_months  = round(runif(1, 1, 12)),
      monthly_charge = pmax(1000, round(rnorm(1, 8000, 1500))),
      data_usage_gb  = round(runif(1, 0.2, 2.0), 2),
      call_minutes   = pmax(10, round(rnorm(1, 80, 30))),
      num_complaints = sample(3:8, 1),
      payment_delay  = sample(2:6, 1),
      churn          = rbinom(1, 1, 0.75)
    )
  } else if (p == "medium") {
    data.frame(
      customer_id    = sprintf("MTN-%04d", i),
      tenure_months  = round(runif(1, 12, 36)),
      monthly_charge = pmax(1000, round(rnorm(1, 14000, 3000))),
      data_usage_gb  = round(runif(1, 2.0, 6.0), 2),
      call_minutes   = pmax(50, round(rnorm(1, 200, 60))),
      num_complaints = sample(1:3, 1),
      payment_delay  = sample(0:3, 1),
      churn          = rbinom(1, 1, 0.35)
    )
  } else {
    data.frame(
      customer_id    = sprintf("MTN-%04d", i),
      tenure_months  = round(runif(1, 36, 72)),
      monthly_charge = pmax(5000, round(rnorm(1, 22000, 4000))),
      data_usage_gb  = round(runif(1, 6.0, 20.0), 2),
      call_minutes   = pmax(100, round(rnorm(1, 380, 80))),
      num_complaints = sample(0:1, 1),
      payment_delay  = sample(0:1, 1),
      churn          = rbinom(1, 1, 0.08)
    )
  }
})

df <- bind_rows(rows)
cat("✅ Dataset simulé :", nrow(df), "clients,", ncol(df), "variables\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PRÉPARATION & EXPLORATION DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

cat("\n──── Aperçu des données ────\n")
glimpse(df)

cat("\n──── Statistiques descriptives ────\n")
summary(df[, -1])   # exclure customer_id

cat("\n──── Valeurs manquantes ────\n")
cat("Total NA :", sum(is.na(df)), "\n")

cat("\n──── Distribution de la cible (churn) ────\n")
table(df$churn) %>% print()
prop.table(table(df$churn)) %>% round(3) %>% print()

# ── Variables numériques pour la modélisation ──
num_vars <- c("tenure_months", "monthly_charge", "data_usage_gb",
              "call_minutes", "num_complaints", "payment_delay")

# ── Standardisation (Z-score) pour le clustering ──
df_scaled <- df %>%
  select(all_of(num_vars)) %>%
  scale() %>%
  as.data.frame()

# ── Matrice de corrélation ──
cor_mat <- cor(df[, num_vars])
cat("\n──── Matrice de corrélation ────\n")
print(round(cor_mat, 2))

# ─────────────────────────────────────────────────────────────────────────────
# 3. SEGMENTATION PAR K-MEANS
# ─────────────────────────────────────────────────────────────────────────────

# ── 3.1  Méthode du coude (Elbow) ──
wss <- sapply(1:10, function(k) {
  kmeans(df_scaled, centers = k, nstart = 25, iter.max = 100)$tot.withinss
})

p_elbow <- ggplot(data.frame(k = 1:10, wss = wss), aes(x = k, y = wss)) +
  geom_line(color = "#FFD700", linewidth = 1.2) +
  geom_point(color = "#FF6B35", size = 3) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "#00A651", linewidth = 0.8) +
  annotate("text", x = 3.3, y = max(wss)*0.85, label = "k = 3 optimal",
           color = "#00A651", fontface = "bold") +
  labs(title = "Méthode du Coude — Sélection du nombre de clusters",
       x = "Nombre de clusters (k)", y = "WSS (inertie intra-cluster)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", color = "#1A1A2E"))

print(p_elbow)

# ── 3.2  Silhouette Score pour k = 2 à 6 ──
sil_scores <- sapply(2:6, function(k) {
  km  <- kmeans(df_scaled, centers = k, nstart = 25)
  sil <- silhouette(km$cluster, dist(df_scaled))
  mean(sil[, 3])
})

cat("\n──── Silhouette scores (k=2..6) ────\n")
print(data.frame(k = 2:6, silhouette = round(sil_scores, 4)))

# ── 3.3  K-Means final : k = 3 ──
set.seed(42)
km_final <- kmeans(df_scaled, centers = 3, nstart = 50, iter.max = 200)
df$cluster <- km_final$cluster

cat("\n──── Distribution des clusters ────\n")
table(df$cluster) %>% print()

# ── 3.4  Profil moyen par cluster ──
cluster_profile <- df %>%
  group_by(cluster) %>%
  summarise(
    n_clients       = n(),
    tenure_moy      = round(mean(tenure_months), 1),
    charge_moy      = round(mean(monthly_charge), 0),
    data_moy        = round(mean(data_usage_gb), 2),
    calls_moy       = round(mean(call_minutes), 0),
    complaints_moy  = round(mean(num_complaints), 2),
    delay_moy       = round(mean(payment_delay), 2),
    taux_churn      = round(mean(churn), 3)
  )

cat("\n──── Profil des clusters ────\n")
print(cluster_profile)

# ── Réétiquetage selon le niveau de risque ──
# Le cluster avec le taux de churn le plus élevé → Cluster 1 (Haut risque)
risk_order <- cluster_profile %>% arrange(desc(taux_churn)) %>% pull(cluster)
label_map  <- setNames(c("Cluster 1 - Haut risque",
                          "Cluster 2 - Risque moyen",
                          "Cluster 3 - Faible risque"),
                       risk_order)
df$segment <- label_map[as.character(df$cluster)]

# ─────────────────────────────────────────────────────────────────────────────
# 4. ANALYSE EN COMPOSANTES PRINCIPALES (PCA)
# ─────────────────────────────────────────────────────────────────────────────

pca_res <- prcomp(df_scaled, center = FALSE, scale. = FALSE)

cat("\n──── Variance expliquée par composante ────\n")
var_exp <- summary(pca_res)$importance[2, ] * 100
print(round(var_exp[1:4], 2))

# ── Visualisation PCA colorée par cluster ──
p_pca <- fviz_pca_ind(
  pca_res,
  geom.ind     = "point",
  col.ind      = as.factor(df$cluster),
  palette      = c("#E74C3C", "#F39C12", "#27AE60"),
  addEllipses  = TRUE,
  ellipse.type = "confidence",
  legend.title = "Cluster",
  title        = "PCA — Visualisation des segments clients MTN Bénin"
) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(p_pca)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PRÉPARATION POUR LA MODÉLISATION PRÉDICTIVE
# ─────────────────────────────────────────────────────────────────────────────

df_model <- df %>%
  select(all_of(num_vars), churn) %>%
  mutate(churn = as.factor(churn))

# ── Split Train / Test (80/20) ──
train_idx  <- createDataPartition(df_model$churn, p = 0.80, list = FALSE)
train_data <- df_model[train_idx, ]
test_data  <- df_model[-train_idx, ]

cat("\n──── Taille Train/Test ────\n")
cat("Train :", nrow(train_data), "| Test :", nrow(test_data), "\n")

# ── Paramètres de cross-validation (5-fold) ──
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Niveaux de la variable cible
levels(train_data$churn) <- c("Non", "Oui")
levels(test_data$churn)  <- c("Non", "Oui")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODÈLE 1 — RÉGRESSION LOGISTIQUE (Baseline)
# ─────────────────────────────────────────────────────────────────────────────

cat("\n══════════════════════════════════════\n")
cat("  MODÈLE 1 : Régression Logistique\n")
cat("══════════════════════════════════════\n")

set.seed(42)
model_lr <- train(
  churn ~ .,
  data      = train_data,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)

# ── Prédictions ──
pred_lr      <- predict(model_lr, test_data)
pred_lr_prob <- predict(model_lr, test_data, type = "prob")[, "Oui"]

# ── Métriques ──
cm_lr  <- confusionMatrix(pred_lr, test_data$churn, positive = "Oui")
roc_lr <- roc(as.numeric(test_data$churn == "Oui"), pred_lr_prob, quiet = TRUE)

cat("Accuracy  :", round(cm_lr$overall["Accuracy"] * 100, 1), "%\n")
cat("AUC-ROC   :", round(auc(roc_lr), 3), "\n")
cat("Sensibilité:", round(cm_lr$byClass["Sensitivity"] * 100, 1), "%\n")
cat("Spécificité:", round(cm_lr$byClass["Specificity"] * 100, 1), "%\n")

# ── Coefficients du modèle ──
cat("\n──── Coefficients ────\n")
coef_lr <- summary(model_lr$finalModel)$coefficients
print(round(coef_lr, 4))

# ─────────────────────────────────────────────────────────────────────────────
# 7. MODÈLE 2 — RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

cat("\n══════════════════════════════════════\n")
cat("  MODÈLE 2 : Random Forest\n")
cat("══════════════════════════════════════\n")

set.seed(42)
model_rf <- train(
  churn ~ .,
  data      = train_data,
  method    = "rf",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = expand.grid(mtry = c(2, 3, 4)),
  ntree     = 300
)

pred_rf      <- predict(model_rf, test_data)
pred_rf_prob <- predict(model_rf, test_data, type = "prob")[, "Oui"]

cm_rf  <- confusionMatrix(pred_rf, test_data$churn, positive = "Oui")
roc_rf <- roc(as.numeric(test_data$churn == "Oui"), pred_rf_prob, quiet = TRUE)

cat("Accuracy  :", round(cm_rf$overall["Accuracy"] * 100, 1), "%\n")
cat("AUC-ROC   :", round(auc(roc_rf), 3), "\n")
cat("Sensibilité:", round(cm_rf$byClass["Sensitivity"] * 100, 1), "%\n")
cat("Spécificité:", round(cm_rf$byClass["Specificity"] * 100, 1), "%\n")

# ── Importance des variables ──
imp_rf <- varImp(model_rf)$importance %>%
  rownames_to_column("variable") %>%
  arrange(desc(Overall))

cat("\n──── Importance des variables (RF) ────\n")
print(imp_rf)

p_imp_rf <- ggplot(imp_rf, aes(x = reorder(variable, Overall), y = Overall)) +
  geom_col(fill = "#27AE60", width = 0.6) +
  coord_flip() +
  labs(title = "Random Forest — Importance des variables",
       x = NULL, y = "Importance (%)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(p_imp_rf)

# ─────────────────────────────────────────────────────────────────────────────
# 8. MODÈLE 3 — XGBOOST (Modèle final optimisé)
# ─────────────────────────────────────────────────────────────────────────────

cat("\n══════════════════════════════════════\n")
cat("  MODÈLE 3 : XGBoost\n")
cat("══════════════════════════════════════\n")

xgb_grid <- expand.grid(
  nrounds          = c(100, 200),
  max_depth        = c(3, 5),
  eta              = c(0.05, 0.1),
  gamma            = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample        = 0.8
)

set.seed(42)
model_xgb <- train(
  churn ~ .,
  data      = train_data,
  method    = "xgbTree",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = xgb_grid,
  verbosity = 0
)

pred_xgb      <- predict(model_xgb, test_data)
pred_xgb_prob <- predict(model_xgb, test_data, type = "prob")[, "Oui"]

cm_xgb  <- confusionMatrix(pred_xgb, test_data$churn, positive = "Oui")
roc_xgb <- roc(as.numeric(test_data$churn == "Oui"), pred_xgb_prob, quiet = TRUE)

cat("Accuracy  :", round(cm_xgb$overall["Accuracy"] * 100, 1), "%\n")
cat("AUC-ROC   :", round(auc(roc_xgb), 3), "\n")
cat("Sensibilité:", round(cm_xgb$byClass["Sensitivity"] * 100, 1), "%\n")
cat("Spécificité:", round(cm_xgb$byClass["Specificity"] * 100, 1), "%\n")

# ── Importance des variables XGBoost ──
imp_xgb <- varImp(model_xgb)$importance %>%
  rownames_to_column("variable") %>%
  arrange(desc(Overall))

cat("\n──── Importance des variables (XGB) ────\n")
print(imp_xgb)

p_imp_xgb <- ggplot(imp_xgb, aes(x = reorder(variable, Overall), y = Overall)) +
  geom_col(fill = "#E74C3C", width = 0.6) +
  coord_flip() +
  labs(title = "XGBoost — Importance des variables (modèle final)",
       x = NULL, y = "Importance (%)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(p_imp_xgb)

# ─────────────────────────────────────────────────────────────────────────────
# 9. COMPARAISON DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

results_df <- data.frame(
  Modele   = c("Logistic Regression", "Random Forest", "XGBoost"),
  Accuracy = c(round(cm_lr$overall["Accuracy"]  * 100, 1),
               round(cm_rf$overall["Accuracy"]  * 100, 1),
               round(cm_xgb$overall["Accuracy"] * 100, 1)),
  AUC_ROC  = c(round(auc(roc_lr), 3),
               round(auc(roc_rf), 3),
               round(auc(roc_xgb), 3))
)

cat("\n══════════════════════════════════════════════════════\n")
cat("  TABLEAU COMPARATIF DES MODÈLES\n")
cat("══════════════════════════════════════════════════════\n")
print(results_df)

# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALISATIONS FINALES
# ─────────────────────────────────────────────────────────────────────────────

# ── 10.1  Courbes ROC des 3 modèles ──
p_roc <- ggplot() +
  # LR
  geom_line(data = data.frame(fpr = 1 - roc_lr$specificities,
                               tpr = roc_lr$sensitivities),
            aes(x = fpr, y = tpr, color = "Logistic Regression"), linewidth = 1) +
  # RF
  geom_line(data = data.frame(fpr = 1 - roc_rf$specificities,
                               tpr = roc_rf$sensitivities),
            aes(x = fpr, y = tpr, color = "Random Forest"), linewidth = 1) +
  # XGBoost
  geom_line(data = data.frame(fpr = 1 - roc_xgb$specificities,
                               tpr = roc_xgb$sensitivities),
            aes(x = fpr, y = tpr, color = "XGBoost"), linewidth = 1.3) +
  geom_abline(linetype = "dashed", color = "grey60") +
  scale_color_manual(
    name   = "Modèle",
    values = c("Logistic Regression" = "#3498DB",
               "Random Forest"       = "#27AE60",
               "XGBoost"             = "#E74C3C")
  ) +
  annotate("text", x = 0.60, y = 0.30,
           label = sprintf("AUC — LR: %.2f | RF: %.2f | XGB: %.2f",
                           auc(roc_lr), auc(roc_rf), auc(roc_xgb)),
           size = 3.5, color = "grey30") +
  labs(title = "Courbes ROC — Comparaison des modèles de churn",
       x = "Taux de faux positifs (1 - Spécificité)",
       y = "Taux de vrais positifs (Sensibilité)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

print(p_roc)

# ── 10.2  Taux de churn par segment ──
churn_seg <- df %>%
  group_by(segment) %>%
  summarise(taux_churn = round(mean(churn) * 100, 1),
            n = n())

couleurs_seg <- c("Cluster 1 - Haut risque"   = "#E74C3C",
                  "Cluster 2 - Risque moyen"   = "#F39C12",
                  "Cluster 3 - Faible risque"  = "#27AE60")

p_churn_seg <- ggplot(churn_seg,
                       aes(x = reorder(segment, -taux_churn),
                           y = taux_churn, fill = segment)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = paste0(taux_churn, "%")),
            vjust = -0.5, fontface = "bold", size = 4.5) +
  scale_fill_manual(values = couleurs_seg) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "Taux de churn par segment client",
       subtitle = "MTN Bénin — Analyse de segmentation",
       x = NULL, y = "Taux de churn (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title    = element_text(face = "bold"),
        axis.text.x   = element_text(angle = 15, hjust = 1))

print(p_churn_seg)

# ── 10.3  Revenu mensuel moyen par segment ──
rev_seg <- df %>%
  group_by(segment) %>%
  summarise(revenu_moy = mean(monthly_charge))

p_rev <- ggplot(rev_seg,
                aes(x = reorder(segment, revenu_moy),
                    y = revenu_moy, fill = segment)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = scales::comma(round(revenu_moy, 0), suffix = " XOF")),
            hjust = -0.1, fontface = "bold", size = 4) +
  coord_flip() +
  scale_fill_manual(values = couleurs_seg) +
  scale_x_discrete() +
  labs(title = "Revenu mensuel moyen par segment",
       x = NULL, y = "Charge mensuelle moyenne (XOF)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(p_rev)

# ── 10.4  Barres comparatives Accuracy & AUC ──
results_long <- results_df %>%
  pivot_longer(cols = c(Accuracy, AUC_ROC), names_to = "Metrique", values_to = "Valeur")

p_compare <- ggplot(results_long, aes(x = Modele, y = Valeur, fill = Metrique)) +
  geom_col(position = "dodge", width = 0.5) +
  geom_text(aes(label = round(Valeur, 2)), position = position_dodge(0.5),
            vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Accuracy" = "#3498DB", "AUC_ROC" = "#E74C3C")) +
  labs(title = "Comparaison des performances des modèles",
       x = NULL, y = "Score", fill = "Métrique") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(p_compare)

# ─────────────────────────────────────────────────────────────────────────────
# 11. SCORE DE RISQUE XGBoost — SYSTÈME D'ALERTE PRÉCOCE
# ─────────────────────────────────────────────────────────────────────────────

df$churn_prob_xgb <- predict(model_xgb, df[, num_vars], type = "prob")[, "Oui"]

df <- df %>%
  mutate(risk_label = case_when(
    churn_prob_xgb >= 0.70 ~ "🔴 Très haut risque",
    churn_prob_xgb >= 0.40 ~ "🟡 Risque modéré",
    TRUE                   ~ "🟢 Faible risque"
  ))

cat("\n══════════════════════════════════════════════════════\n")
cat("  SYSTÈME D'ALERTE PRÉCOCE — Distribution\n")
cat("══════════════════════════════════════════════════════\n")
table(df$risk_label) %>% print()

# ── Top 10 clients à risque maximum ──
cat("\n──── Top 10 clients à risque de churn le plus élevé ────\n")
df %>%
  arrange(desc(churn_prob_xgb)) %>%
  select(customer_id, tenure_months, monthly_charge,
         num_complaints, payment_delay, churn_prob_xgb, risk_label) %>%
  head(10) %>%
  print()

# ─────────────────────────────────────────────────────────────────────────────
# 12. RECOMMANDATIONS BUSINESS
# ─────────────────────────────────────────────────────────────────────────────

cat("\n")
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║        RECOMMANDATIONS BUSINESS — MTN BÉNIN                 ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n\n")

cat("1. 🔴 CLUSTER 1 — RÉTENTION URGENTE\n")
cat("   → Offres de fidélité ciblées (remise 20-30% sur renouvellement)\n")
cat("   → Programme de résolution accélérée des réclamations (< 24h)\n")
cat("   → Appels proactifs au service client pour les clients à très haut risque\n\n")

cat("2. ⚡ SYSTÈME D'ALERTE PRÉCOCE (XGBoost)\n")
cat("   → Déployer le score mensuel sur l'ensemble des abonnés actifs\n")
cat("   → Déclencher automatiquement des actions CRM pour P(churn) > 0.70\n")
cat("   → Révision trimestrielle du modèle pour maintenir la performance\n\n")

cat("3. 📞 AMÉLIORATION DU SERVICE CLIENT\n")
cat("   → Variable num_complaints = principal prédicteur de churn\n")
cat("   → Réduire le délai moyen de résolution des plaintes\n")
cat("   → Mettre en place un NPS mensuel par segment\n\n")

cat("4. 📶 UPSELL CLUSTER 2 — FIDÉLISATION\n")
cat("   → Proposer des forfaits data supérieurs avant la fenêtre de churn (12-24 mois)\n")
cat("   → Offres bundle voix + data pour augmenter le revenu et la stickiness\n\n")

cat("5. 💰 OPTIMISATION DU REVENU\n")
cluster_profile %>%
  left_join(
    df %>% group_by(cluster) %>% summarise(
      revenu_total_mois = sum(monthly_charge),
      perte_estimee     = sum(monthly_charge[churn == 1])
    ),
    by = "cluster"
  ) %>%
  select(cluster, n_clients, taux_churn, revenu_total_mois, perte_estimee) %>%
  print()

cat("\n══════════════════════════════════════════════════════════════\n")
cat("  ✅ Analyse complète MTN Bénin terminée avec succès !\n")
cat("══════════════════════════════════════════════════════════════\n")
