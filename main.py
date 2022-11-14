
from __future__ import annotations

# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re  # regular expressions
import chess
from typing import Optional

# custom modules
import chess_const

# download full file here https://drive.google.com/file/d/0Bw0y3jV73lx_aXE3RnhmeE5Rb1E/edit?usp=sharing
FILE_PATH_CHESS_DATA = 'data/sample_10k_chess_games.txt'

CHUNK_SIZE = int(1000)

STATS_TO_TRACK = [
    'avg turns in game',
    'ranks pushed ahead', 'ranks pushed back',
    'files pushed', 'dist pushed',
    'captures', 'been captured'
]

NON_REQUIRED_COLUMNS = ['#', 't']  # '#' declares header row and 't' is the index, both can be removed for this analysis

TESTING = True  # set to False for production


def parse_game_meta_data_string_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split string column with misc game dataframe 'df' into separate columns
    Apply correct column names
    Sort by date
    Convert string columns with bool values to bool
    df : input dataframe
    return : parsed dataframe
    """

    column_names = list(df.columns)[-1].split()  # get column names as list from last column
    column_names = _get_column_names_wo_column_number(column_names)
    column_names = [column_name for column_name in column_names if column_name not in NON_REQUIRED_COLUMNS]

    df = df.reset_index()
    df = df.dropna(axis=1, how='all')  # remove any nan columns

    df = pd.concat(
        [df.iloc[:, 0].str.split(expand=True), df.iloc[:, 1]],
        axis=1
    )  # resulting dataframe has unwanted data column with linear row indexes
    df = _drop_columns_with_linear_index(df)
    df = df.set_axis(column_names, axis=1)  # rename columns

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date  # removes default time on the clock
    df = df.sort_values(by='date', ignore_index=True)  # sorts, but keeps linear index

    df = df.applymap(bool_keyword_to_bool)  # remove the column name from boolean values

    return df


def _get_column_names_wo_column_number(column_names_raw: list[str]) -> list[str]:
    """
    Removes column numbers in format {number}{.} from column names
    column_names_raw : list of numbered names
    return : list of names without numbers
    """
    column_names_clean = []
    for column_name in column_names_raw:
        column_names_clean.append(re.sub(r'\d*\.+', '', column_name))  # remove column number and dots
    return column_names_clean


def _drop_columns_with_linear_index(df: pd.DataFrame) -> pd.DataFrame:
    """"
    Removes columns with linear index from 1 to len(df) from dataframe
    """
    for column in df.columns:
        if df[column].astype('string').equals(
                pd.Series(range(1, len(df) + 1)).astype('string')
        ):
            df = df.drop(column, axis=1)
    return df


def convert_move_column_to_standard_format(df: pd.DataFrame, orig_column_name: str) -> pd.DataFrame:
    """
    Transforms strings containing all game moves in the format
    {side_to_move}{turn_number}{.}{san} to {san}
    and splits the moves into list
    df : dataframe with one column containing strings of moves in SAN format
    orig_column_name : original name of column
    return : data frame with move column in SAN format as lists
    """

    standard_column_name = 'moves_as_san'  # portable game notation
    df[orig_column_name] = df[orig_column_name].apply(
        lambda x: re.sub(r'[WB]\d+\.', '', x).split() if isinstance(x, str) else None
    )  # removes the {side_to_move}{turn_number}{.} from the moves string and splits it into list

    df = df.rename(columns={orig_column_name: standard_column_name})
    return df


def bool_keyword_to_bool(value: str):
    """
    value : String that potentially contains 'true' or 'false'
    return :
        value converted to boolean if it contained the corresponding keyword
        value if not
    """
    if isinstance(value, str):
        if 'true' in value.lower():
            return True
        elif 'false' in value.lower():
            return False
        else:
            return value
    else:
        return value


def get_game_stats(moves_as_san: pd.Series) -> tuple[list[chess.Board], dict[str, pd.DataFrame]]:
    """
    Plays games and collects usage statistics for different piece types
    Games with illegal or invalid moves are returned as missing values
    moves_as_san : series of lists containing game moves in SAN format
    return :
        boards : list of boards in final state
        stats_per_game : dictionary of dataframes with usage stats (keys) for the piece types (columns)
    """

    boards = list()

    stats_per_game = dict()
    for stat_to_track in STATS_TO_TRACK:
        stats_per_game[stat_to_track] = pd.DataFrame(columns=chess_const.PIECE_NAMES)

    print("Playing games...")
    for _, game_as_san in moves_as_san.items():
        board, stats_avg = _play_game_from_san_list(game_as_san)

        boards.append(board)
        for stat_to_track in STATS_TO_TRACK:
            stats_per_game[stat_to_track] = pd.concat(
                [stats_per_game[stat_to_track], stats_avg[stat_to_track].to_frame().T],
                ignore_index=True
            )

    return boards, stats_per_game


def _play_game_from_san_list(game_as_san: list) -> tuple[Optional[chess.Board], dict[str, pd.Series]]:
    """
    board : chess.Board
    san_list :
        List of strings representing moves in SAN format
    return :
        board
            Updated board with all moves in san_list put onto the move stack
            None if an exception was raised
        stats_avg
            Dictionary of series containing the average stats of one game per piece type
            Dictionary of series of None values if an exception was raised
    exceptions :
        If there is an illegal, ambiguous or invalid move (raised as ValueError)
        If parsing a move from SAN raises an unknown exception
        If a move was detected as a capture, but there was no piece on the target field
    """

    stats_per_turn, stats_avg, stats_avg_blank_row, board = _initialize_data_structs_game_stats(len(game_as_san))

    # play game and collect stats per turn
    for turn_number in range(len(game_as_san)):
        san = game_as_san[turn_number]
        try:
            move = board.parse_san(san)
        except Exception as e:
            if isinstance(e, ValueError):  # chess lib treats invalid moves as ValueError
                print(f"Move caused ValueError '{e}' for SAN move '{san}'")
            else:
                print(f"Unknown exception {e} for SAN move '{san}'")

            print("Stopped playing game due to error in SAN.")
            return None, stats_avg_blank_row

        stats_per_turn = _collect_move_stats_per_piece_type(board, move, stats_per_turn)
        if stats_per_turn is None:
            return None, stats_avg_blank_row

        board.push(move)

    # calculate cumulative stats for all turns and divide by number of starting instances of piece --> average
    for stat_to_track in STATS_TO_TRACK:
        stats_avg[stat_to_track] = \
            stats_per_turn[stat_to_track].sum().truediv(chess_const.STARTING_INSTANCES_PER_PIECE_TYPE)

    return board, stats_avg


def _initialize_data_structs_game_stats(num_of_turns: int) -> tuple[
        dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, pd.Series], chess.Board]:

    # initialize stats data structs as dicts
    stats_per_turn = dict()
    stats_avg = dict()

    for stat_to_track in STATS_TO_TRACK:
        stats_per_turn[stat_to_track] = pd.DataFrame(  # dict of dataframes with stats per turn
            0,
            columns=chess_const.PIECE_NAMES,
            index=np.arange(num_of_turns)
        )

    stats_avg_blank_row = dict.fromkeys(  # returned if there is an exception in the game
        STATS_TO_TRACK,
        pd.Series(None, index=chess_const.PIECE_NAMES, dtype=object)
    )

    board = chess.Board()  # initialize board
    return stats_per_turn, stats_avg, stats_avg_blank_row, board


def _collect_move_stats_per_piece_type(
        board: chess.Board,
        move: chess.Move,
        stats_per_turn: dict[str, pd.DataFrame]) -> Optional[dict[str, pd.DataFrame]]:

    piece_type_pushed = board.piece_type_at(move.from_square)

    files_pushed = chess.square_file(move.from_square) - chess.square_file(move.to_square)

    if board.turn:  # white's turn
        ranks_pushed = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    else:
        ranks_pushed = chess.square_rank(move.from_square) - chess.square_rank(move.to_square)

    for piece_type in chess_const.PIECE_TYPES:
        piece_name = chess_const.PIECE_NAMES[piece_type - 1]

        halfturn_number = len(board.move_stack)

        # number of turns in the game
        instances_of_piece = 0
        for color in chess_const.COLORS:
            instances_of_piece += len(board.pieces(piece_type, color))
        stats_per_turn['avg turns in game'][piece_name].iat[halfturn_number] = instances_of_piece

        if piece_type == piece_type_pushed:
            # ranks pushed
            if ranks_pushed >= 0:
                stats_per_turn['ranks pushed ahead'][piece_name].iat[halfturn_number] = ranks_pushed
            else:
                stats_per_turn['ranks pushed back'][piece_name].iat[halfturn_number] = np.abs(ranks_pushed)

            # files pushed
            stats_per_turn['files pushed'][piece_name].iat[halfturn_number] = np.abs(files_pushed)

            # distance pushed
            stats_per_turn['dist pushed'][piece_name].iat[halfturn_number] = abs(ranks_pushed) + abs(files_pushed)

            if board.is_capture(move):
                # piece type at target square
                if board.piece_type_at(move.to_square):
                    captured_piece_name = chess.piece_name(board.piece_type_at(move.to_square))
                else:
                    print(f"{move} was detected as a capture, but there is no piece on the target field. "
                          f"Stopped playing game.")
                    return None
                # for pushed piece type
                stats_per_turn['captures'][piece_name].iat[halfturn_number] = 1
                # for captured piece type
                stats_per_turn['been captured'][captured_piece_name].iat[halfturn_number] = 1
    return stats_per_turn


def print_md(df: pd.DataFrame) -> None:
    df_head = df.head()
    print(f'Data frame with {len(df)} rows')
    print(df_head.to_markdown(tablefmt='grid'))


def dropna_print_dropped_count(df: pd.DataFrame) -> pd.DataFrame:
    na_rows_to_drop = len(df.index[df.isna().any(axis=1)])
    df = df.dropna()
    print(f"Removed {na_rows_to_drop} row(s) due to missing values.")
    return df


def plot_a_stat(stats_per_game: dict[str, pd.DataFrame], stat_to_plot: str) -> None:
    sns.histplot(stats_per_game[stat_to_plot], multiple="stack")
    plt.title = stat_to_plot
    plt.legend(title=stat_to_plot, labels=chess_const.PIECE_NAMES)
    plt.show()
    plt.interactive(False)


def main():
    print(f"Reading file '{FILE_PATH_CHESS_DATA}'...")
    chunk_reader = pd.read_csv(
        FILE_PATH_CHESS_DATA,
        skiprows=4,
        sep='###',
        engine='python',
        chunksize=CHUNK_SIZE)

    n_rows_to_process = CHUNK_SIZE

    if TESTING:
        games = pd.DataFrame(chunk_reader.get_chunk(n_rows_to_process))  # only work with read chunk (for testing)
    else:
        games = pd.concat(chunk_reader)  # for production

    print("...done")

    games = parse_game_meta_data_string_to_columns(games)

    games = convert_move_column_to_standard_format(games, orig_column_name='game')

    games = dropna_print_dropped_count(games)

    print_md(games)

    boards, stats_per_game = get_game_stats(games['moves_as_san'])

    games['boards'] = boards
    games = dropna_print_dropped_count(games)

    print_md(games)

    for stat_to_track in STATS_TO_TRACK:
        stats_per_game[stat_to_track] = stats_per_game[stat_to_track].dropna()  # remove invalid games
        print(f"Showing {stat_to_track} per figure type:")
        print_md(stats_per_game[stat_to_track])

    plot_a_stat(stats_per_game, 'captures')


if __name__ == "__main__":
    main()
