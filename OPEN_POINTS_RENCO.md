# Punti da chiarire con Renco

Checklist di decisioni aperte sul generatore MDR. Aggiornare quando Renco risponde.

---

## Scheduling (date e predecessori)

### 1. Predecessori senza durata timeline

**Comportamento attuale:** se un predecessore è presente nell’MDR ma non ha `duration_days` dalla timeline reconciliation, non ritarda il successore — il suo `finish` coincide con lo `start`, quindi non sposta la data di inizio del documento dipendente.

**Visibilità debug:** sul successore, `DBG_Flags` include `pred_no_duration` e `DBG_PredFinishes` annota i pred con `[no_duration]`. Sul predecessore resta il flag `no_duration`.

**Da chiedere:** è accettabile, oppure un predecessore senza durata dovrebbe comunque bloccare il successore (es. usando una durata minima di default, o segnalando errore)?

**Riferimento codice:** `mdr_generator/5_schedule.py` — calcolo `pred_finish_pairs` e `finish_by_key`.

---

### 2. Cicli nei predecessori RACI (dati)

**Problema:** nel catalogo `raci_matrix.DocumentPredecessors` esistono dipendenze circolari (es. `equipment list` ↔ `equipment summary`). Un grafo con cicli non ammette un ordine topologico valido.

**Da chiedere:** quale arco del ciclo va rimosso o invertito nel RACI? Serve correzione dei dati sorgente.

**Evidenza:** flag `cycle` in colonna `DBG_Flags` (se `schedule.debug_columns = true`); audit in `output/.../schedule_audit.json` → `cycle_audit`.

---

### 3. Politica di fallback quando resta un ciclo

**Comportamento attuale** (dopo il sort topologico sui nodi aciclici):

1. I nodi coinvolti nel ciclo vengono aggiunti in coda in **ordine alfabetico** per `TitleKey`.
2. Per ogni documento, entrano nel calcolo solo i predecessori **già processati** (`finish_by_key`).
3. Il predecessore ciclico che viene **dopo** in ordine alfabetico viene di fatto **ignorato** per il nodo processato per primo.
4. **Nessun documento viene escluso** dal calcolo date; le date risultano **asimmetriche** rispetto al ciclo.

**Esempio:** `equipment list` → `equipment summary` (ciclo bidirezionale)

| Documento           | Ordine fallback | Effetto                                      |
|---------------------|-----------------|----------------------------------------------|
| `equipment list`    | 1° (alfabetico) | Non aspetta `equipment summary`              |
| `equipment summary` | 2°              | Aspetta il finish di `equipment list`        |

**Da chiedere:** va bene questo fallback, oppure preferiscono:

- errore / blocco generazione MDR se c’è un ciclo;
- scelta esplicita di quale arco ignorare (non alfabetico);
- altro criterio (es. priorità per disciplina/capitolo).

**Riferimento codice:** `mdr_generator/5_schedule.py` — `_topological_order`, `_schedule_line_items`.

---

## Decisioni già prese (non in attesa)

| Argomento              | Decisione                          |
|------------------------|------------------------------------|
| Separatore titolo MDR  | Solo `\|` per suffissi 3b e 3d     |
| Lingua suffissi titoli | Inglese (prompt 3b/3d)             |
| Colonne debug schedule | `schedule.debug_columns` in settings |

---

*Ultimo aggiornamento: 2026-07-27*
