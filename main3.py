import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from fpdf import FPDF
import tempfile

# -----------------------------------------------------------------------------
# 1. CONFIGURAZIONE PAGINA E STILE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Gestione Scorte EOQ & ROP", layout="wide", page_icon="📦")

st.title("📦 Sistema di Ottimizzazione Inventario (EOQ & ROP)")
st.markdown("""
Questo software supporta le decisioni di approvvigionamento per materiali a domanda indipendente.
Calcola il **Lotto Economico (EOQ)** e il **Punto di Riordino (ROP)** con **Scorta di Sicurezza**, 
simulando le performance su un orizzonte di **3 anni**.
""")

# -----------------------------------------------------------------------------
# 2. INPUT PARAMETRI (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Parametri Economici")
    D = st.number_input("Domanda Annua (Unità)", min_value=100, value=2400, step=100, help="D: Domanda totale prevista per l'anno")
    S = st.number_input("Costo Setup/Ordine (€)", min_value=5.0, value=50.0, step=5.0, help="S: Costo fisso per emettere un ordine")
    H = st.number_input("Costo Mantenimento (€/unità/anno)", min_value=0.10, value=2.0, step=0.1, help="H: Costo per tenere un'unità a magazzino per un anno")
    C = st.number_input("Costo Unitario Articolo (€)", value=10.0, step=1.0)
    
    st.header("2. Parametri Logistici & Variabilità")
    L = st.number_input("Lead Time Medio (Giorni)", min_value=1, value=10, step=1, help="L: Tempo medio tra ordine e consegna")
    # Parametri per la simulazione realistica e il calcolo SS
    st.info("Parametri per gestire l'incertezza:")
    L_std = st.number_input("Dev. Std. Lead Time (Giorni)", min_value=0.25, value=2.0, step=0.25, help="Variabilità dei tempi di consegna del fornitore")
    D_std = st.number_input("Dev. Std. Domanda Giornaliera", min_value=0.25, value=2.0, step=0.25, help="Variabilità della domanda giornaliera")
    
    st.header("3. Livello di Servizio")
    livello_servizio_pct = st.slider("Target Service Level (%)", 80.0, 99.0, 95.0, step=0.5, help="la probabilità di essere in grado " \
    "di soddisfare la domanda dei clienti senza affrontare alcun backorder o vendita persa.")
    livello_servizio = livello_servizio_pct / 100.0

# -----------------------------------------------------------------------------
# 3. MOTORE DI CALCOLO (MODEL)
# -----------------------------------------------------------------------------
def calcola_kpi(D, S, H, L, L_std, D_std, livello_servizio):
    # EOQ = sqrt(2DS/H)
    if H == 0: H = 0.0001 # Protezione divisione per zero
    eoq = np.sqrt((2 * D * S) / H)
    
    # Domanda Media Giornaliera
    d_avg = D / 365.0
    
    # Calcolo Z-score
    Z = norm.ppf(livello_servizio)
    
    # Calcolo Scorta di Sicurezza (SS)
    # Sigma Combinato durante il Lead Time
    varianza_LT = (L * (D_std**2)) + ((d_avg**2) * (L_std**2))
    sigma_LT = np.sqrt(varianza_LT)
    
    ss = Z * sigma_LT
    
    # Calcolo ROP
    lead_time_demand = d_avg * L
    rop = lead_time_demand + ss
    
    # Costi Totali Stimati (Teorici)
    costo_ordinazione = (D / eoq) * S
    costo_mantenimento = (eoq / 2) * H + (ss * H) # Include mantenimento SS
    costo_totale = costo_ordinazione + costo_mantenimento
    
    return {
        "EOQ": round(eoq, 0),
        "SS": round(ss, 0),
        "ROP": round(rop, 0),
        "D_avg": d_avg,
        "Costo_Totale": costo_totale
    }

kpi = calcola_kpi(D, S, H, L, L_std, D_std, livello_servizio)

# -----------------------------------------------------------------------------
# 4. MOTORE DI SIMULAZIONE (3 ANNI)
# -----------------------------------------------------------------------------
@st.cache_data
def esegui_simulazione_avanzata(eoq, rop, ss, d_avg, D_std, L, L_std, anni=3):
    giorni = 365 * anni
    inventario = [eoq + ss] # 
    ordini = []  #  Inizializzazione lista vuota
    
    # Generazione Domanda Sintetica con Variabilità
    # Aggiungiamo un leggero trend e stagionalità per realismo
    t = np.arange(giorni)
    trend = t * (d_avg * 0.0001) # Leggero trend di crescita
    stagionalita = np.sin(2 * np.pi * t / 365) * (d_avg * 0.2) # +/- 20% stagionalità
    
    # Domanda base + variabilità casuale + trend + stagionalità
    rumore = np.random.normal(0, D_std, giorni)
    domanda_sim = d_avg + trend + stagionalita + rumore
    domanda_sim = np.maximum(domanda_sim, 0) # No domanda negativa
    
    storico_inv = [] # Inizializzazione lista vuota
    storico_ordini_emessi = [] # Inizializzazione lista vuota
    
    for giorno in range(giorni):
        # 1. Arrivi merce
        arrivi = sum([q for d, q in ordini if d == giorno])
        # Accesso solo all'ultimo valore dell'inventario 
        scorta_attuale = inventario[-1] + arrivi
        
        # 2. Consumo
        cons = domanda_sim[giorno]
        scorta_attuale -= cons
        
        # Gestione Backorder (stock negativo permesso per calcolo costi shortage)
        storico_inv.append(scorta_attuale)
        inventario.append(scorta_attuale)
        
        # 3. Riordino
        # Posizione = Scorta Fisica + Ordini Pendenti
        pendenti = sum([q for d, q in ordini if d > giorno])
        posizione = scorta_attuale + pendenti
        
        if posizione <= rop:
            # Emetti ordine
            lt_effettivo = int(np.random.normal(L, L_std))
            lt_effettivo = max(1, lt_effettivo)
            giorno_arrivo = giorno + lt_effettivo
            
            # Evitiamo ordini oltre l'orizzonte temporale per pulizia
            if giorno_arrivo < giorni:
                ordini.append((giorno_arrivo, eoq))
                storico_ordini_emessi.append(giorno)

    return pd.DataFrame({
        "Giorno": range(giorni),
        "Inventario": storico_inv,
        "Domanda": domanda_sim,
        "ROP": [rop]*giorni,
        "SafetyStock": [ss]*giorni
    })

# Pulsante per rigenerare la simulazione (nuovi numeri casuali)
if st.sidebar.button("🔄 Rigenera Simulazione"):
    st.cache_data.clear()

# Creazione dataframe pandas usando la funzione precedentemente definita
df_sim = esegui_simulazione_avanzata(kpi['EOQ'], kpi['ROP'], kpi['SS'], kpi['D_avg'], D_std, L, L_std)

# -----------------------------------------------------------------------------
# 5. VISUALIZZAZIONE KPI E GRAFICI (VIEW)
# -----------------------------------------------------------------------------
# Riga KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lotto Economico (EOQ)", f"{int(kpi['EOQ'])} unità", help="Quantità ottimale per ordine")
# FIXED: Accessed specific keys (kpi['ROP'], etc) instead of the whole dict
c2.metric("Punto di Riordino (ROP)", f"{int(kpi['ROP'])} unità", help="Livello scorta che fa scattare l'ordine")
c3.metric("Scorta Sicurezza", f"{int(kpi['SS'])} unità", help="Buffer contro la variabilità")
c4.metric("Costo Totale Annuo Est.", f"€ {kpi['Costo_Totale']:,.2f}", help="Escluso costo acquisto merce")

# Grafico Interattivo
st.subheader(f"Simulazione Dinamica Inventario ({len(df_sim)} giorni)")
st.markdown("Il grafico mostra l'andamento 'a dente di sega' realistico, influenzato da variabilità e ritardi.")

fig = go.Figure()

# Linea Inventario
fig.add_trace(go.Scatter(
    x=df_sim['Giorno'], y=df_sim['Inventario'],
    mode='lines', name='Livello Inventario',
    line=dict(color='#1f77b4', width=1.5),
    fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)' # Area sottesa
))

# Linea ROP
# Per il corretto funzionamento e visualizzazione va specificata la singola colonna del dataframe simulazione
fig.add_trace(go.Scatter(
    x=df_sim['Giorno'], y=df_sim['ROP'],
    mode='lines', name='Punto di Riordino (ROP)',
    line=dict(color='#ff7f0e', width=2, dash='dash')
))

# Linea Safety Stock
fig.add_trace(go.Scatter(
    x=df_sim['Giorno'], y=df_sim['SafetyStock'],
    mode='lines', name='Scorta di Sicurezza',
    line=dict(color='#d62728', width=2, dash='dot')
))

fig.update_layout(
    xaxis_title="Giorno (Orizzonte 3 Anni)",
    yaxis_title="Unità in Magazzino",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, width="stretch")

# Dati grezzi
with st.expander("Visualizza Dati Tabellari Giornalieri"):
    st.dataframe(df_sim)

# -----------------------------------------------------------------------------
# 6. EXPORT PDF ED EXCEL
# -----------------------------------------------------------------------------
st.markdown("### Esportazione Dati e Report")

def crea_report_pdf(kpi, df, fig):
    pdf = FPDF()
    pdf.add_page()
    
    # Titolo
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Report Analisi Inventario", 0, 1, 'C')
    
    # Sezione Parametri
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Risultati Ottimizzazione:", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"EOQ (Lotto Economico): {int(kpi['EOQ'])} unita'", 0, 1)
    # Accesso alle chiavi specifiche per il report PDF
    pdf.cell(0, 8, f"ROP (Punto di Riordino): {int(kpi['ROP'])} unita'", 0, 1)
    pdf.cell(0, 8, f"Scorta di Sicurezza: {int(kpi['SS'])} unita'", 0, 1)
    pdf.cell(0, 8, f"Costo Totale Annuo Stimato: Euro {kpi['Costo_Totale']:,.2f}", 0, 1)
    pdf.ln(5)
    
    # Statistiche Simulazione
    min_inv = df['Inventario'].min()
    stockouts = (df['Inventario'] < 0).sum()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Statistiche Simulazione (3 Anni):", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Livello Minimo Raggiunto: {int(min_inv)} unita'", 0, 1)
    pdf.cell(0, 8, f"Giorni di Stockout Totali: {stockouts}", 0, 1)
    
    # Inserimento Immagine Grafico
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            # Nota: Kaleido è richiesto per l'esportazione delle imagini statiche ma non è stato implementato
            fig.write_image(tmp.name, width=800, height=400)
            pdf.image(tmp.name, x=10, y=100, w=190)
    except Exception as e:
        pdf.cell(0, 10, "Impossibile generare immagine grafico (Richiede 'kaleido')", 0, 1)

    return pdf.output(dest='S').encode('latin-1')

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    # Generazione PDF
    try:
        pdf_bytes = crea_report_pdf(kpi, df_sim, fig)
        st.download_button(
            label="📄 Scarica Report PDF",
            data=pdf_bytes,
            file_name="report_inventario.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Errore generazione PDF: {e}")

with col_dl2:
    # Export CSV
    csv_data = df_sim.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Scarica Dati Excel/CSV",
        data=csv_data,
        file_name="simulazione_inventario_3anni.csv",
        mime="text/csv"
    )