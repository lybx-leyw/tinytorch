#include <stdio.h>
#include <stdlib.h>

static char board[9];
static int results[19683];
static int idx = 0;
static int check_win(char p) 
{
    int wins[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
    for (int i = 0; i < 8; i++)
        if (board[wins[i][0]] == p && board[wins[i][1]] == p && board[wins[i][2]] == p)
            return 1;
    return 0;
}
static int full() 
{
    for (int i = 0; i < 9; i++) if (board[i] == '.') return 0;
    return 1;
}
static void save() 
{
    FILE* f = fopen("tictactoe.txt", "a");
    for (int i = 0; i < 9; i++) fputc(board[i], f);
    if (check_win('X')) fputs(" P\n", f);
    else if (check_win('O')) fputs(" N\n", f);
    else if (full()) fputs(" M\n", f);
    else fputs(" Q\n", f);
    fclose(f);
}
static void dfs(int turn) 
{
    if (check_win('X') || check_win('O') || full()) {
        save();
        return;
    }
    for (int i = 0; i < 9; i++) {
        if (board[i] == '.') {
            board[i] = (turn == 0) ? 'X' : 'O';
            dfs(!turn);
            board[i] = '.';
        }
    }
}

int create_data() 
{
    for (int i = 0; i < 9; i++) board[i] = '.';
    FILE* f =fopen("tictactoe.txt", "w");
    fclose(f);
    dfs(0);
    return 0;
}