import sklearn.utils.validation
import sklearn.utils
import sys

_original_check_array = sklearn.utils.validation.check_array

def _fixed_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs.pop('force_all_finite')
    return _original_check_array(*args, **kwargs)

sklearn.utils.validation.check_array = _fixed_check_array
sklearn.utils.check_array = _fixed_check_array

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

pd.set_option("display.max_columns", None)
np.set_printoptions(suppress=True, precision=4)


FILE_PATH = os.path.join('data_in', 'Spotify-2000.csv')
print("--- 1. INCARCARE DATE ---")

if not os.path.exists(FILE_PATH):
    print(f"EROARE: fisier negasit {FILE_PATH}")
    sys.exit(1)

df_raw = pd.read_csv(FILE_PATH)

cols_numerice = [
    'Beats Per Minute (BPM)', 'Energy', 'Danceability',
    'Loudness (dB)', 'Liveness', 'Valence',
    'Acousticness', 'Speechiness'
]

X = df_raw[cols_numerice].copy()

for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].astype(str).str.replace(',', '.', regex=False)
        X[c] = pd.to_numeric(X[c], errors='coerce')

X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(numeric_only=True), inplace=True)

X.index = (df_raw['Title'].astype(str) + " - " + df_raw['Artist'].astype(str)).values
X.index.name = "Melodie"

scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
n, m = X_std.shape

print("\n--- 2. RULARE ACP ---")
os.makedirs('data_out/ACP', exist_ok=True)
os.makedirs('graphics/ACP', exist_ok=True)

R = np.corrcoef(X_std.values, rowvar=False)
R_df = pd.DataFrame(R, index=cols_numerice, columns=cols_numerice).rename_axis('Variabila')

val_proprii, vec_proprii = np.linalg.eigh(R)

idx = np.argsort(val_proprii)[::-1]
alpha = val_proprii[idx].real
vec_proprii = vec_proprii[:, idx].real

C_np = X_std.values @ vec_proprii
C_df = pd.DataFrame(C_np, index=X_std.index, columns=[f'C{i+1}' for i in range(m)]).rename_axis('Melodie')

R_xc = vec_proprii * np.sqrt(alpha)
R_xc_df = pd.DataFrame(R_xc, index=cols_numerice, columns=[f'C{i+1}' for i in range(m)]).rename_axis('Variabila')

Comm = np.cumsum(R_xc**2, axis=1)
Comm_df = pd.DataFrame(Comm, index=cols_numerice, columns=[f'C{i+1}' for i in range(m)]).rename_axis('Variabila')

Cosin = R_xc**2
Cosin_df = pd.DataFrame(Cosin, index=cols_numerice, columns=[f'C{i+1}' for i in range(m)]).rename_axis('Variabila')

Contrib = (R_xc**2 / alpha) * 100
Contrib_df = pd.DataFrame(Contrib, index=cols_numerice, columns=[f'C{i+1}' for i in range(m)]).rename_axis('Variabila')

tabel_varianta = pd.DataFrame({
    'Valoare Proprie': alpha,
    'Varianta (%)': alpha / np.sum(alpha) * 100,
    'Varianta Cumulata (%)': np.cumsum(alpha / np.sum(alpha) * 100)
}, index=[f'C{i+1}' for i in range(m)]).rename_axis('Componenta')

R_df.to_csv('data_out/ACP/Matrice_Corelatie.csv', float_format='%.6f')
tabel_varianta.to_csv('data_out/ACP/Distributia_Variantei.csv', float_format='%.6f')

C_df.to_csv('data_out/ACP/Scoruri_Componente.csv', float_format='%.6f')
R_xc_df.to_csv('data_out/ACP/Corelatii_Factoriale.csv', float_format='%.6f')
Comm_df.to_csv('data_out/ACP/Comunalitati.csv', float_format='%.6f')
Cosin_df.to_csv('data_out/ACP/Cosinusuri.csv', float_format='%.6f')
Contrib_df.to_csv('data_out/ACP/Contributii.csv', float_format='%.6f')

print("Tabelele si scorurile componentelor au fost salvate.")

plt.figure(figsize=(12, 8))
C_df_plot = C_df.copy()
C_df_plot['Year'] = df_raw['Year'].values

scatter = sns.scatterplot(data=C_df_plot, x='C1', y='C2', hue='Year', palette='viridis', alpha=0.7)

plt.title('Harta Scorurilor ACP - Colorata dupa Anul lansarii')
plt.xlabel('Componenta 1 (C1)')
plt.ylabel('Componenta 2 (C2)')
plt.grid(True, alpha=0.3)
plt.legend(title='An Lansare', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('graphics/ACP/Harta_Scoruri_An.png', dpi=300)
plt.close()
print("Harta scorurilor a fost generata.")


plt.figure(figsize=(10, 6))
plt.plot(range(1, m + 1), alpha, 'o-')
plt.axhline(1, color='r', linestyle='--', label='Kaiser (λ=1)')
plt.xticks(range(1, m + 1))
plt.title('Scree Plot (ACP)')
plt.xlabel('Componenta')
plt.ylabel('Valoare proprie')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphics/ACP/Plot_Varianta.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(C_df['C1'], C_df['C2'], alpha=0.5)
plt.title('Plot Scoruri ACP (C1 vs C2)')
plt.xlabel('Componenta 1 (C1)')
plt.ylabel('Componenta 2 (C2)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphics/ACP/Plot_Scoruri_C1_C2.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), color='gray')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.axvline(0, color='k', linestyle='--', linewidth=1)

for var in cols_numerice:
    x = R_xc_df.loc[var, 'C1']
    y = R_xc_df.loc[var, 'C2']
    plt.arrow(0, 0, x, y, alpha=0.8, head_width=0.03, length_includes_head=True)
    plt.text(x * 1.08, y * 1.08, var, ha='center', va='center')

plt.title('Cercul Corelatiilor (ACP)')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('graphics/ACP/Cercul_Corelatiilor.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(R_xc_df, annot=True, center=0, fmt=".2f")
plt.title('Corelograma Corelatii Variabile–Componente (ACP)')
plt.tight_layout()
plt.savefig('graphics/ACP/Corelograma_Corelatii.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(Comm_df, annot=True, vmin=0, vmax=1, fmt=".2f")
plt.title('Corelograma Comunalitati (ACP)')
plt.tight_layout()
plt.savefig('graphics/ACP/Corelograma_Comunalitati.png', dpi=300)
plt.close()

print("Graficele ACP au fost salvate in folderul graphics.")

print("\n--- 3. RULARE ANALIZA FACTORIALA ---")
os.makedirs('data_out/Fact', exist_ok=True)
os.makedirs('graphics/Fact', exist_ok=True)

chi, p_val = calculate_bartlett_sphericity(X_std)
kmo_all, kmo_model = calculate_kmo(X_std)

print(f"Bartlett p-value: {p_val:.6f}")
print(f"KMO Global: {kmo_model:.4f}")

if p_val >= 0.05:
    print("\nAtentie: Testul Bartlett a esuat, analiza factoriala nu poate continua.")
    sys.exit(0)

kmo_df = pd.DataFrame({'KMO': kmo_all}, index=cols_numerice).rename_axis('Variabila')
kmo_df.to_csv('data_out/Fact/KMO_Variabile.csv', float_format='%.6f')

plt.figure(figsize=(6, 8))
sns.heatmap(kmo_df, annot=True, vmin=0, vmax=1, fmt=".2f")
plt.title(f'Corelograma Indici KMO (Global: {kmo_model:.3f})')
plt.tight_layout()
plt.savefig('graphics/Fact/Corelograma_KMO.png', dpi=300)
plt.close()

nr_factori = int(np.sum(alpha > 1))
print(f"Numar factori selectati (Kaiser): {nr_factori}")

fa_rot = FactorAnalyzer(n_factors=nr_factori, rotation='varimax')
fa_rot.fit(X_std)

load_rot = pd.DataFrame(fa_rot.loadings_, index=cols_numerice,
                        columns=[f'F{i+1}' for i in range(nr_factori)]).rename_axis('Variabila')
comm_rot = pd.DataFrame(fa_rot.get_communalities(), index=cols_numerice, columns=['Comm']).rename_axis('Variabila')
scores_rot = pd.DataFrame(fa_rot.transform(X_std), index=X_std.index,
                          columns=[f'F{i+1}' for i in range(nr_factori)]).rename_axis('Melodie')
var_rot = pd.DataFrame(np.array(fa_rot.get_factor_variance()),
                       index=['SS Loadings', 'Prop Var', 'Cum Var'],
                       columns=[f'F{i+1}' for i in range(nr_factori)]).rename_axis('Metrica')

load_rot.to_csv('data_out/Fact/Loadings_Varimax.csv', float_format='%.6f')
comm_rot.to_csv('data_out/Fact/Comunalitati_Varimax.csv', float_format='%.6f')
scores_rot.to_csv('data_out/Fact/Scoruri_Varimax.csv', float_format='%.6f')
var_rot.to_csv('data_out/Fact/Varianta_Varimax.csv', float_format='%.6f')

plt.figure(figsize=(8, 6))
sns.heatmap(load_rot, annot=True, center=0, fmt=".2f")
plt.title('Heatmap Loadings (Varimax)')
plt.tight_layout()
plt.savefig('graphics/Fact/Heatmap_Loadings_Varimax.png', dpi=300)
plt.close()

if nr_factori >= 2:
    plt.figure(figsize=(10, 6))
    plt.scatter(scores_rot['F1'], scores_rot['F2'], alpha=0.5)
    plt.title('Plot Scoruri Factoriale (Varimax: F1 vs F2)')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphics/Fact/Plot_Scoruri_Varimax.png', dpi=300)
    plt.close()

fa_none = FactorAnalyzer(n_factors=nr_factori, rotation=None)
fa_none.fit(X_std)

load_none = pd.DataFrame(fa_none.loadings_, index=cols_numerice,
                         columns=[f'F{i+1}' for i in range(nr_factori)]).rename_axis('Variabila')
comm_none = pd.DataFrame(fa_none.get_communalities(), index=cols_numerice, columns=['Comm']).rename_axis('Variabila')
scores_none = pd.DataFrame(fa_none.transform(X_std), index=X_std.index,
                           columns=[f'F{i+1}' for i in range(nr_factori)]).rename_axis('Melodie')
var_none = pd.DataFrame(np.array(fa_none.get_factor_variance()),
                        index=['SS Loadings', 'Prop Var', 'Cum Var'],
                        columns=[f'F{i+1}' for i in range(nr_factori)]).rename_axis('Metrica')

load_none.to_csv('data_out/Fact/Loadings_Nerotit.csv', float_format='%.6f')
comm_none.to_csv('data_out/Fact/Comunalitati_Nerotit.csv', float_format='%.6f')
scores_none.to_csv('data_out/Fact/Scoruri_Nerotit.csv', float_format='%.6f')
var_none.to_csv('data_out/Fact/Varianta_Nerotit.csv', float_format='%.6f')

plt.figure(figsize=(8, 6))
sns.heatmap(load_none, annot=True, center=0, fmt=".2f")
plt.title('Heatmap Loadings (Fara rotatie)')
plt.tight_layout()
plt.savefig('graphics/Fact/Heatmap_Loadings_Nerotit.png', dpi=300)
plt.close()

if nr_factori >= 2:
    plt.figure(figsize=(10, 6))
    plt.scatter(scores_none['F1'], scores_none['F2'], alpha=0.5)
    plt.title('Plot Scoruri Factoriale (Fara rotatie: F1 vs F2)')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphics/Fact/Plot_Scoruri_Nerotit.png', dpi=300)
    plt.close()

plt.figure(figsize=(6, 8))
sns.heatmap(comm_rot, annot=True, vmin=0, vmax=1, fmt=".2f")
plt.title('Corelograma Comunalitati (Analiza Factoriala - Varimax)')
plt.tight_layout()
plt.savefig('graphics/Fact/Corelograma_Comunalitati.png', dpi=300)
plt.close()

print("\nAnaliza s-a incheiat si rezultatele au fost salvate.")
