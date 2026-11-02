"""
Descompactação dos arquivos .dbc do DATASUS em Python puro (qualquer versão).

Um .dbc é um DBF comprimido: o cabeçalho do DBF vem sem compressão, seguido de
4 bytes de CRC e do corpo comprimido pelo algoritmo PKWare DCL "implode". Este
módulo implementa o "explode" correspondente, seguindo o blast.c de Mark Adler
(zlib/contrib/blast), e devolve o DBF completo.

Se o pacote datasus-dbc estiver instalado (binário, bem mais rápido), ele é usado.
"""

import struct

try:
    import datasus_dbc as _datasus_dbc
except ImportError:
    _datasus_dbc = None

# Tabelas do formato (blast.c), em forma compacta: cada byte é
# (repetições - 1) << 4 | comprimento do código.
_LITLEN = [11, 124, 8, 7, 28, 7, 188, 13, 76, 4, 10, 8, 12, 10, 12, 10, 8, 23, 8,
           9, 7, 6, 7, 8, 7, 6, 55, 8, 23, 24, 12, 11, 7, 9, 11, 12, 6, 7, 22, 5,
           7, 24, 6, 11, 9, 6, 7, 22, 7, 11, 38, 7, 9, 8, 25, 11, 8, 11, 9, 12,
           8, 12, 5, 38, 5, 38, 5, 11, 7, 5, 6, 21, 6, 10, 53, 8, 7, 24, 10, 27,
           44, 253, 253, 253, 252, 252, 252, 13, 12, 45, 12, 45, 12, 61, 12, 45,
           44, 173]
_LENLEN = [2, 35, 36, 53, 38, 23]
_DISTLEN = [2, 20, 53, 230, 247, 151, 248]
_BASE = [3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40, 72, 136, 264]
_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def _tabela(compacta):
    """
    Monta a tabela de decodificação de um código de Huffman canônico do formato:
    indexada pelos próximos `maximo` bits do buffer (lidos a partir do menos
    significativo), devolve (símbolo, bits consumidos). No formato, os bits dos
    códigos vêm invertidos.
    """
    comprimentos = []
    for byte in compacta:
        comprimentos += [byte & 15] * ((byte >> 4) + 1)
    maximo = max(comprimentos)

    tabela = [None] * (1 << maximo)
    codigo = 0
    for comprimento in range(1, maximo + 1):
        for simbolo, c in enumerate(comprimentos):
            if c != comprimento:
                continue
            # bit i lido do fluxo = bit (comprimento-1-i) do código, invertido
            lidos = 0
            for i in range(comprimento):
                bit = ((codigo >> (comprimento - 1 - i)) & 1) ^ 1
                lidos |= bit << i
            for alto in range(1 << (maximo - comprimento)):
                tabela[lidos | (alto << comprimento)] = (simbolo, comprimento)
            codigo += 1
        codigo <<= 1
    return tabela, maximo


_TAB_LIT, _MAX_LIT = _tabela(_LITLEN)
_TAB_LEN, _MAX_LEN = _tabela(_LENLEN)
_TAB_DIST, _MAX_DIST = _tabela(_DISTLEN)


def explodir(dados):
    """Descomprime um fluxo PKWare DCL implode (bytes) e devolve os bytes originais."""
    if len(dados) < 2:
        raise ValueError("fluxo implode truncado")
    literais_codificados, bits_dicionario = dados[0], dados[1]
    if literais_codificados > 1 or not 4 <= bits_dicionario <= 6:
        raise ValueError("cabeçalho implode inválido")

    saida = bytearray()
    buffer, n_bits, pos, fim = 0, 0, 2, len(dados)
    tab_lit, max_lit = _TAB_LIT, _MAX_LIT
    tab_len, max_len = _TAB_LEN, _MAX_LEN
    tab_dist, max_dist = _TAB_DIST, _MAX_DIST
    base, extra = _BASE, _EXTRA

    while True:
        # garante bits suficientes para o maior passo (1 + 13 + 8 + ... < 32)
        while n_bits < 32 and pos < fim:
            buffer |= dados[pos] << n_bits
            pos += 1
            n_bits += 8

        flag = buffer & 1
        buffer >>= 1
        n_bits -= 1

        if flag:                                        # par comprimento/distância
            simbolo, c = tab_len[buffer & ((1 << max_len) - 1)]
            buffer >>= c
            n_bits -= c
            e = extra[simbolo]
            comprimento = base[simbolo] + (buffer & ((1 << e) - 1))
            buffer >>= e
            n_bits -= e
            if comprimento == 519:                      # fim do fluxo
                break

            while n_bits < 24 and pos < fim:
                buffer |= dados[pos] << n_bits
                pos += 1
                n_bits += 8
            simbolo, c = tab_dist[buffer & ((1 << max_dist) - 1)]
            buffer >>= c
            n_bits -= c
            d = 2 if comprimento == 2 else bits_dicionario
            distancia = (simbolo << d) + (buffer & ((1 << d) - 1)) + 1
            buffer >>= d
            n_bits -= d

            inicio = len(saida) - distancia
            if inicio < 0:
                raise ValueError("distância além do início dos dados")
            if distancia >= comprimento:
                saida += saida[inicio:inicio + comprimento]
            else:                                       # cópia sobreposta: repete o padrão
                padrao = saida[inicio:]
                repeticoes, resto = divmod(comprimento, distancia)
                saida += padrao * repeticoes + padrao[:resto]
        else:                                           # literal
            if literais_codificados:
                simbolo, c = tab_lit[buffer & ((1 << max_lit) - 1)]
                buffer >>= c
                n_bits -= c
            else:
                simbolo = buffer & 0xFF
                buffer >>= 8
                n_bits -= 8
            saida.append(simbolo)

        if n_bits < 0:
            raise ValueError("fluxo implode truncado")
    return bytes(saida)


def descompactar(conteudo):
    """Conteúdo de um .dbc (bytes) -> conteúdo do DBF (bytes)."""
    if _datasus_dbc is not None:
        return _datasus_dbc.decompress_bytes(conteudo)
    tam_cabecalho = struct.unpack("<H", conteudo[8:10])[0]
    return conteudo[:tam_cabecalho] + explodir(conteudo[tam_cabecalho + 4:])
