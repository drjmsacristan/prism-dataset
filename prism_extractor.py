import lief
import numpy as np

COMMON_NAMES = {
    '.text', '.data', '.rdata', '.bss', '.rsrc', '.reloc',
    '.idata', '.edata', '.pdata', '.tls', '.debug',
    'CODE', 'DATA', '.ndata', '.CRT', '.sxdata'
}

N_MAX = 16
F_REDUCED = 20

def extract_section_features(section):
    """Extrae features de una seccion PE individual."""

    # 1. Name encoding (8 dims) - hash del nombre
    name = section.name.strip('\x00')
    name_hash = np.zeros(8, dtype=np.float32)
    for i, c in enumerate(name[:8]):
        name_hash[i] = ord(c) / 127.0

    # 2. Sizes (3 dims)
    raw_size = section.size
    virt_size = section.virtual_size if section.virtual_size > 0 else 1
    ratio = raw_size / virt_size if virt_size > 0 else 0.0
    sizes = np.array([
        np.log1p(raw_size) / 25.0,
        np.log1p(virt_size) / 25.0,
        min(ratio, 10.0) / 10.0
    ], dtype=np.float32)

    # 3. Permissions (6 dims)
    chars = section.characteristics
    perms = np.array([
        1.0 if chars & 0x40000000 else 0.0,  # MEM_READ
        1.0 if chars & 0x80000000 else 0.0,  # MEM_WRITE
        1.0 if chars & 0x20000000 else 0.0,  # MEM_EXECUTE
        1.0 if chars & 0x02000000 else 0.0,  # MEM_DISCARDABLE
        1.0 if chars & 0x00000020 else 0.0,  # CNT_CODE
        1.0 if chars & 0x00000040 else 0.0,  # CNT_INITIALIZED_DATA
    ], dtype=np.float32)

    # 4. Byte entropy (1 dim)
    content = bytes(section.content)
    if len(content) > 0:
        counts = np.bincount(np.frombuffer(content, dtype=np.uint8), minlength=256)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs))) / 8.0
    else:
        entropy = 0.0
    entropy_arr = np.array([entropy], dtype=np.float32)

    # 5. Entropy quartiles (4 dims)
    if len(content) >= 256:
        windows = [content[i:i+256] for i in range(0, len(content)-256, 256)]
        ents = []
        for w in windows[:100]:
            c = np.bincount(np.frombuffer(w, dtype=np.uint8), minlength=256)
            p = c / c.sum()
            p = p[p > 0]
            ents.append(-np.sum(p * np.log2(p)) / 8.0)
        ents = np.array(ents)
        quartiles = np.percentile(ents, [25, 50, 75, 100]).astype(np.float32)
    else:
        quartiles = np.zeros(4, dtype=np.float32)

    # 6. Position index (1 dim) - se rellena despues
    position = np.array([0.0], dtype=np.float32)

    # 7. Anomaly flags (3 dims)
    wx = 1.0 if (perms[1] == 1.0 and perms[2] == 1.0) else 0.0
    unusual_name = 1.0 if name not in COMMON_NAMES else 0.0
    zero_raw = 1.0 if raw_size == 0 else 0.0
    anomaly = np.array([unusual_name, wx, zero_raw], dtype=np.float32)

    # Concatenar version reducida (sin histograma de bytes)
    row = np.concatenate([
        name_hash,    # 8
        sizes,        # 3
        perms,        # 6
        entropy_arr,  # 1
        quartiles,    # 4
        position,     # 1 (placeholder)
        anomaly       # 3
    ])               # Total: 26 dims (reducido sin histograma)

    return row, name, content


def extract_prism_matrix(filepath, n_max=N_MAX):
    """
    Extrae la matriz PRISM de un fichero PE.
    Devuelve (M, mask, n_secciones) o None si falla el parseo.
    """
    try:
        pe = lief.parse(str(filepath))
        if pe is None:
            return None

        sections = list(pe.sections)
        n_real = len(sections)

        # Dimensiones: n_max filas de seccion + 1 fila global
        F = 26
        M = np.zeros((n_max + 1, F), dtype=np.float32)
        mask = np.zeros(n_max + 1, dtype=np.uint8)

        # Rellenar filas de secciones
        n_fill = min(n_real, n_max)
        for i in range(n_fill):
            row, name, content = extract_section_features(sections[i])
            row[18] = i / n_max  # position index normalizado
            M[i] = row
            mask[i] = 1

        # Fila global simplificada (ultima fila)
        n_imports = 0
        n_exports = 0
        try:
            if pe.has_imports:
                n_imports = sum(len(lib.entries) for lib in pe.imports)
            if pe.has_exports:
                n_exports = len(pe.get_export().entries)
        except:
            pass

        global_row = np.zeros(F, dtype=np.float32)
        global_row[0] = min(n_real, 255) / 255.0
        global_row[1] = np.log1p(n_imports) / 10.0
        global_row[2] = np.log1p(n_exports) / 10.0
        global_row[3] = 1.0 if pe.has_signatures else 0.0
        global_row[4] = 1.0 if pe.has_resources else 0.0

        M[n_max] = global_row
        mask[n_max] = 1

        return M, mask, n_real

    except Exception as e:
        return None


if __name__ == '__main__':
    import sys, os

    # Test con un binario del sistema
    test_files = [
        '/usr/bin/ls',
        '/usr/bin/python3',
        '/usr/bin/grep',
    ]

    print('=== Test del extractor PRISM ===')
    print(f'N_MAX={N_MAX}, F={F_REDUCED} (reducido)')
    print()

    for f in test_files:
        if os.path.exists(f):
            result = extract_prism_matrix(f)
            if result is not None:
                M, mask, n_sec = result
                print(f'Fichero: {f}')
                print(f'  Secciones reales: {n_sec}')
                print(f'  Shape matriz M: {M.shape}')
                print(f'  Filas activas (mask=1): {mask.sum()}')
                print(f'  Entropia seccion 0: {M[0][18]:.4f}')
                print()
            else:
                print(f'{f}: no es PE valido (esperado en Linux)')
