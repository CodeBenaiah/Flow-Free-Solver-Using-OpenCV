#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <time.h>

using namespace std;

inline namespace FileIO
{
    void setIn(string s) { freopen(s.c_str(), "r", stdin); }
    void setOut(string s) { freopen(s.c_str(), "w", stdout); }
}

vector<vector<int>> vis;
int totcol = 0;
int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};
bool f1(int ccol, int x, int y, int ccnt, vector<vector<int>> &g, vector<int> &col,  vector<pair<int,int>> &st, vector<pair<int,int>> &ed)
{
    if(ccol == ccnt){
        if(totcol == g.size() * g[0].size())
            return true;
        else return false;
    }

    totcol++;
    vis[x][y] = 1;
    int prev = g[x][y];
    g[x][y] = col[ccol];

    if(x == ed[ccol].first && y == ed[ccol].second){
        if(f1(ccol + 1, st[ccol + 1].first, st[ccol + 1].second, ccnt, g, col, st, ed)) return true;
        else {
            totcol--;
            vis[x][y] = 0;
            g[x][y] = prev;
            return false;
        }
    }

    for(int i = 0; i<4; i++)
    {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if(nx >= 0 && nx<g.size() && ny >= 0 && ny < g[0].size() && vis[nx][ny] == 0 &&  (g[nx][ny] == 0 || g[nx][ny] == col[ccol])){
            if(f1(ccol, nx, ny, ccnt, g, col, st, ed)) return true;
        }
    }

    totcol--;
    vis[x][y] = 0;
    g[x][y] = prev;

    return false;
}

int main()
{
    // clock_t start = clock();
    setIn("board.txt");
    setOut("output.txt");
    int N;
    cin>>N;
    vis = vector<vector<int>>(N, vector<int>(N));

    vector<vector<int>> grid(N, vector<int>(N));
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++) cin>>grid[i][j];
    }

    map<int, vector<pair<int, int>>> mp;
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            if(grid[i][j] == 0) continue;
            mp[grid[i][j]].push_back({i, j});
        }
    }

    vector<int> col;
    vector<pair<int,int>> st, ed;
    for(auto it : mp)
    {
        int cur_col = it.first;
        auto [si, sj] = it.second[0];
        auto [di, dj] = it.second[1];

        col.push_back(cur_col);
        st.push_back({si, sj});
        ed.push_back({di, dj});
    }

    int cnt = col.size();

    if(f1(0, st[0].first, st[0].second, cnt, grid, col, st, ed)){
        for(int i = 0; i<N; i++){
            for(int j = 0; j<N; j++){
                if(i == N-1 && j == N-1)
                    cout<<grid[i][j];
                else cout<<grid[i][j]<<' ';
            }
        }
    }   
    else {
        cout<<"Answer not possible"<<endl;
    }
    // clock_t end = clock();
    // double elapsed = double(end - start) / CLOCKS_PER_SEC;
    // cout<<elapsed<<endl;
}