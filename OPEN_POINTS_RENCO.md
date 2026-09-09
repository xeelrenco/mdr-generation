# Punti da chiarire con Renco

Checklist di decisioni aperte sul generatore MDR. Aggiornare quando Renco risponde.

---

## Scheduling (date e predecessori)

### 1. Predecessori senza durata timeline

**Comportamento attuale:** se un predecessore è presente nell’MDR ma non ha `duration_days` dalla timeline reconciliation, non ritarda il successore — il suo `finish` coincide con lo `start`, quindi non sposta la data di inizio del documento dipendente.

**Visibilità debug:** sul successore, `DBG_Flags` include `pred_no_duration` e `DBG_PredFinishes` annota i pred con `[no_duration]`. Sul predecessore resta il flag `no_duration`.

**Da chiedere:** è accettabile, oppure un predecessore senza durata dovrebbe comunque bloccare il successore (es. usando una durata minima di default, o segnalando errore)?

**Riferimento codice:** `mdr_generator/12_schedule.py` — calcolo `pred_finish_pairs` e `finish_by_key`.

---

### 2. Cicli nei predecessori RACI (dati) — deciso

**Decisione Renco 2026-09-09:** prima `equipment summary`, poi `equipment list`. List dipende da Summary.

Arco da eliminare nel catalogo: `equipment summary` ← `equipment list`.
Arco da tenere: `equipment list` ← `equipment summary`.

**Pipeline:** `12_schedule.py` ignora l’arco inverso (`dropped_cycle_edge`) anche se la riga è ancora in `DocumentPredecessors`, così lo scheduling non dipende dal fallback alfabetico.

**Evidenza:** `schedule_audit.json` → `dropped_predecessor_edges`; flag `cycle` solo se resta un ciclo diverso.

---

### 3. Politica di fallback quando resta un ciclo

Il fallback alfabetico resta **solo** per cicli sconosciuti. Non deve più applicarsi alla coppia Equipment List / Summary.

**Comportamento residuo** (dopo il sort topologico sui nodi aciclici):

1. I nodi coinvolti in un ciclo non corretto vengono aggiunti in coda in **ordine alfabetico** per `TitleKey`.
2. Per ogni documento, entrano nel calcolo solo i predecessori **già processati** (`finish_by_key`).
3. **Nessun documento viene escluso** dal calcolo date.

**Riferimento codice:** `mdr_generator/12_schedule.py` — `_DROPPED_PREDECESSOR_EDGES`, `_topological_order`.

---

## Decisioni già prese (non in attesa)

| Argomento              | Decisione                          |
|------------------------|------------------------------------|
| Separatore titolo MDR  | Solo `\|` per suffissi 3b e 3d     |
| Lingua suffissi titoli | Inglese (prompt 3b/3d)             |
| Colonne debug schedule | `schedule.debug_columns` in settings |
| Ciclo List ↔ Summary   | Summary prima; List dipende da Summary |

---

*Ultimo aggiornamento: 2026-09-09*
