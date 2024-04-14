#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>
using namespace std;

// Namespace for file input/output operations
namespace FileIO {
    void setInputFile(const char* fileName) {
        freopen(fileName, "r", stdin);
    }

    void setOutputFile(const string& fileName) {
        freopen(fileName.c_str(), "w", stdout);
    }
}

// 2D vector to keep track of visited cells
vector<vector<bool>> visited;

// Total number of cells filled
int totalCellsFilled = 0;

// Movement directions (row, col)
const int dx[4] = {0, 0, 1, -1};
const int dy[4] = {1, -1, 0, 0};

// Helper function to solve the grid
bool solvePuzzle(int currentColor, int row, int col, int colorCount, vector<vector<int>>& grid, const vector<int>& colors, const vector<pair<int, int>>& startPositions, const vector<pair<int, int>>& endPositions) {
    if (currentColor == colorCount) {
        // All colors have been placed
        return totalCellsFilled == grid.size() * grid[0].size();
    }

    totalCellsFilled++;
    visited[row][col] = true;
    int prevValue = grid[row][col];
    grid[row][col] = colors[currentColor];

    if (row == endPositions[currentColor].first && col == endPositions[currentColor].second) {
        // Current cell is the end position for the current color
        if (solvePuzzle(currentColor + 1, startPositions[currentColor + 1].first, startPositions[currentColor + 1].second, colorCount, grid, colors, startPositions, endPositions)) {
            return true;
        }
    } else {
        // Try placing the current color in adjacent cells
        for (int i = 0; i < 4; i++) {
            int newRow = row + dx[i];
            int newCol = col + dy[i];
            if (newRow >= 0 && newRow < grid.size() && newCol >= 0 && newCol < grid[0].size() && !visited[newRow][newCol] && (grid[newRow][newCol] == 0 || grid[newRow][newCol] == colors[currentColor])) {
                if (solvePuzzle(currentColor, newRow, newCol, colorCount, grid, colors, startPositions, endPositions)) {
                    return true;
                }
            }
        }
    }

    // Backtrack: Restore the previous value and mark the cell as unvisited
    totalCellsFilled--;
    visited[row][col] = false;
    grid[row][col] = prevValue;
    return false;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: <path_to_input_file>" << endl;
        return -1;
    }

    FileIO::setInputFile(argv[1]);
    FileIO::setOutputFile("./dataset/solution_matrices/output.txt");

    int rows, cols;
    cin >> rows >> cols;
    visited = vector<vector<bool>>(rows, vector<bool>(cols, false));
    vector<vector<int>> grid(rows, vector<int>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cin >> grid[i][j];
        }
    }

    // Map to store the start and end positions for each color
    map<int, vector<pair<int, int>>> colorPositions;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] != 0) {
                colorPositions[grid[i][j]].push_back({i, j});
            }
        }
    }

    vector<int> colors;
    vector<pair<int, int>> startPositions, endPositions;
    for (const auto& [color, positions] : colorPositions) {
        colors.push_back(color);
        startPositions.push_back(positions[0]);
        endPositions.push_back(positions[1]);
    }

    int colorCount = colors.size();

    if (solvePuzzle(0, startPositions[0].first, startPositions[0].second, colorCount, grid, colors, startPositions, endPositions)) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == rows - 1 && j == cols - 1) {
                    cout << grid[i][j];
                } else {
                    cout << grid[i][j] << ' ';
                }
            }
        }
    } else {
        cout << "Answer not possible" << endl;
    }

    return 0;
}