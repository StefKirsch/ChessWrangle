# chess_const.py

"""
This module defines project-level chess constants that are not easily available in the chess library.
"""

import chess

FILES = chess.FILE_NAMES  # columns, numbers 1 to 8
RANKS = [int(i) for i in chess.RANK_NAMES]  # rows, letters a to h

COLORS = [chess.WHITE, chess.BLACK]  # false, true

PIECE_TYPES = [  # list of piece types as integers from 1 to 6
    chess.PAWN, chess.KNIGHT,
    chess.BISHOP, chess.ROOK,
    chess.QUEEN, chess.KING
]

PIECE_NAMES = []  # list of piece names
for piece_type in PIECE_TYPES:
    PIECE_NAMES.append(chess.piece_name(piece_type))

STARTING_INSTANCES_PER_PIECE_TYPE = [2*x for x in [8, 2, 2, 2, 1, 1]]  # for both colors