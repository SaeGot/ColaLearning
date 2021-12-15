#pragma once
#include <fstream>
#include <string>
#include <vector>


using namespace std;

class FileManager
{
public:
	FileManager(string file_Name);
	~FileManager();

private:
	vector<vector<double>> data;
};

