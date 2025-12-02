import os
import numpy as np
import pandas as pd
import time
from .otimizador_aco import OtimizadorACO

class OtimizadorACOInstrumentado(OtimizadorACO):
    """
    Versão do ACO que coleta snapshots da matriz de feromônios por geração.
    - Config:
        ACO_PARAMS.capturar_feromonio: bool (default True)
        ACO_PARAMS.freq_snapshot: int (default 1) -> a cada N gerações
        ACO_PARAMS.salvar_dir: str (default '../graficos' relat.)
        ACO_PARAMS.salvar_prefixo: str (default 'aco_feromonio')
        ACO_PARAMS.salvar_final_npy: bool (default True)
        ACO_PARAMS.salvar_stats_csv: bool (default True)
        ACO_PARAMS.salvar_amostra_csv: bool (default False) -> salva pequeno subset
        ACO_PARAMS.amostra_max: int (default 2000) -> nº máx. de células na amostra CSV
    """
    def __init__(self, config: dict):
        super().__init__(config)
        p = self.config.get("ACO_PARAMS", {})
        self.capturar_feromonio = p.get("capturar_feromonio", True)
        self.freq_snapshot = int(p.get("freq_snapshot", 1))
        self.salvar_dir = p.get("salvar_dir", os.path.join("..", "graficos"))
        self.salvar_prefixo = p.get("salvar_prefixo", "aco_feromonio")
        self.salvar_final_npy = bool(p.get("salvar_final_npy", True))
        self.salvar_stats_csv = bool(p.get("salvar_stats_csv", True))
        self.salvar_amostra_csv = bool(p.get("salvar_amostra_csv", False))
        self.amostra_max = int(p.get("amostra_max", 2000))

        # Histórico leve (stats por geração) e, opcionalmente, amostras
        self.feromonio_stats = []        # [{geracao, min, max, mean, std}, ...]
        self.feromonio_amostras = []     # [(geracao, np.ndarray_amostra), ...]
        self._start_time = None

    def _snapshot_feromonio(self, geracao: int):
        if not self.capturar_feromonio or self.arr_feromonio is None:
            return

        arr = self.arr_feromonio
        # Coleta estatísticas globais
        stats = {
            "geracao": geracao + 1,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "num_profs": int(self.num_profs),
            "num_discs": int(self.num_discs),
            "tempo_decorrido_s": time.time() - self._start_time if self._start_time else None
        }
        self.feromonio_stats.append(stats)

        # Amostra pequena para inspecionar evolução local sem arquivos gigantes
        if self.salvar_amostra_csv:
            n = arr.size
            k = min(self.amostra_max, n)
            # amostra aleatória de k células (uniforme)
            flat = arr.ravel()
            idxs = np.random.choice(n, size=k, replace=False)
            amostra = flat[idxs]
            self.feromonio_amostras.append((geracao + 1, amostra))

    def _atualizar_feromonio_numpy(self, solucoes_geracao):
        # usa a versão do pai para evaporar e depositar
        super()._atualizar_feromonio_numpy(solucoes_geracao)

        # snapshot conforme frequência
        if self.capturar_feromonio and len(self.metricas_iteracao) % self.freq_snapshot == 0:
            self._snapshot_feromonio(geracao=len(self.metricas_iteracao) - 1)

    def _resolver_nucleo(self, callback_iteracao=None):
        # inicia cronômetro
        self._start_time = time.time()

        resultado = super()._resolver_nucleo(callback_iteracao=callback_iteracao)

        # Salvar artefatos (feromônio final e stats)
        try:
            os.makedirs(self.salvar_dir, exist_ok=True)
            base = os.path.join(self.salvar_dir, self.salvar_prefixo)

            # 1) Feromônio final (.npy) para heatmap futuro
            if self.salvar_final_npy and self.arr_feromonio is not None:
                np.save(f"{base}_final.npy", self.arr_feromonio)

            # 2) Estatísticas por geração (.csv)
            if self.salvar_stats_csv and self.feromonio_stats:
                pd.DataFrame(self.feromonio_stats).to_csv(f"{base}_stats.csv", index=False)

            # 3) Amostras por geração (.csv longo porém leve)
            if self.salvar_amostra_csv and self.feromonio_amostras:
                # explode amostras: cada linha = uma célula amostrada (geracao, valor)
                rows = []
                for gen, arr_sample in self.feromonio_amostras:
                    rows.extend([{"geracao": gen, "valor": float(v)} for v in arr_sample])
                pd.DataFrame(rows).to_csv(f"{base}_amostras.csv", index=False)
        except Exception as e:
            # não falha a otimização se I/O falhar
            pass

        # inclui no resultado para inspeção programática
        if resultado is not None:
            resultado["feromonio_stats"] = self.feromonio_stats
            resultado["feromonio_final_shape"] = (int(self.num_profs), int(self.num_discs))
            resultado["feromonio_final_path"] = (
                os.path.join(self.salvar_dir, f"{self.salvar_prefixo}_final.npy")
                if self.salvar_final_npy else None
            )
        return resultado