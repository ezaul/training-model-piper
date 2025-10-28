import os
import glob
import shutil
import warnings
from joblib import Parallel, delayed
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper_timestamped as whisper
import torch # Importação necessária para verificar a GPU

# Ignora avisos, úteis para evitar poluição do console
warnings.filterwarnings("ignore")

# Configurações do processamento
DURACAO_MAXIMA_MS = 10000  # 10 segundos
DURACAO_MINIMA_MS = 5000   # 5 segundos
PASTA_SAIDA = "wav_processados"
NUM_CORES = os.cpu_count()  # Detecta o número de cores da CPU para paralelização

# Carrega o modelo Whisper uma única vez para uso em múltiplos processos
# Use a verificação de GPU da biblioteca torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELO_WHISPER = whisper.load_model("medium", device=DEVICE)

# Função para cortar e salvar os segmentos de áudio
def cortar_e_salvar_segmentos(caminho_arquivo, duracao_maxima_ms, duracao_minima_ms, pasta_saida):
    """
    Corta um arquivo de áudio WAV em segmentos, respeitando pausas.
    """
    
    try:
        audio = AudioSegment.from_wav(caminho_arquivo)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return []

    # Divide o áudio em pedaços baseados no silêncio
    segmentos_fala = split_on_silence(audio,
                                      min_silence_len=500,
                                      silence_thresh=audio.dBFS - 16,
                                      keep_silence=200)

    segmentos_combinados = []
    segmento_atual = AudioSegment.empty()

    for segmento in segmentos_fala:
        if len(segmento_atual) + len(segmento) <= duracao_maxima_ms:
            segmento_atual += segmento
        else:
            if len(segmento_atual) >= duracao_minima_ms:
                segmentos_combinados.append(segmento_atual)
            segmento_atual = segmento
    
    if len(segmento_atual) >= duracao_minima_ms:
        segmentos_combinados.append(segmento_atual)

    # Salva os segmentos temporariamente
    arquivos_temporarios = []
    nome_base = os.path.splitext(os.path.basename(caminho_arquivo))[0]
    
    for i, segmento in enumerate(segmentos_combinados):
        nome_temp = f"temp_{nome_base}_segmento_{i+1}.wav"
        caminho_temp = os.path.join(pasta_saida, nome_temp)
        segmento.export(caminho_temp, format="wav")
        arquivos_temporarios.append(caminho_temp)
    
    return arquivos_temporarios

# Função de validação usando Whisper para verificar cortes de palavra
def validar_e_mover_arquivo(caminho_temp, nome_original, pasta_saida):
    """
    Valida um segmento de áudio e o move se for considerado válido.
    """
    try:
        audio_whisper = whisper.load_audio(caminho_temp)
        # O modelo é passado como argumento
        resultado = whisper.transcribe(MODELO_WHISPER, audio_whisper, language="pt")

        texto_completo = resultado['text'].strip().lower()

        # Condição de validação: Se a transcrição não for vazia
        if not texto_completo:
             os.remove(caminho_temp)
             return

        nome_final = f"{nome_original}_segmento_{os.path.basename(caminho_temp).split('_')[-1]}"
        caminho_final = os.path.join(pasta_saida, nome_final)
        shutil.move(caminho_temp, caminho_final)
        print(f"Arquivo válido movido: {nome_final}")

    except Exception as e:
        print(f"Possível corte de palavra ou erro detectado em {os.path.basename(caminho_temp)}. Deletando...")
        os.remove(caminho_temp)

# Função principal que gerencia o processamento paralelo
def processar_e_validar_audios():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    arquivos_wav = glob.glob("*.wav")
    if not arquivos_wav:
        print("Nenhum arquivo .wav encontrado na pasta.")
        return

    print(f"Iniciando processamento de {len(arquivos_wav)} arquivo(s) usando {NUM_CORES} cores.")
    print("Isso pode levar alguns minutos...")

    # Processamento paralelo com joblib
    arquivos_temporarios_nested = Parallel(n_jobs=NUM_CORES)(
        delayed(cortar_e_salvar_segmentos)(arq, DURACAO_MAXIMA_MS, DURACAO_MINIMA_MS, PASTA_SAIDA) for arq in arquivos_wav
    )
    
    # Validação sequencial dos arquivos gerados para evitar conflitos na GPU
    print("\nIniciando validação e finalização dos arquivos...")
    arquivos_temporarios_flattened = [item for sublist in arquivos_temporarios_nested for item in sublist]
    for temp_file in arquivos_temporarios_flattened:
        nome_original = "_".join(os.path.basename(temp_file).split('_')[1:-2])
        validar_e_mover_arquivo(temp_file, nome_original, PASTA_SAIDA)

    print("\nProcessamento concluído.")

# Executar a função principal
if __name__ == "__main__":
    processar_e_validar_audios()
