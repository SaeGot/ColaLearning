#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

class FileManager
{
public:
	FileManager(string file_Name);
	~FileManager();

private:
	vector<vector<string>> data;
	vector<string> columnName;
};

