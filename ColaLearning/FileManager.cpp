#include "FileManager.h"


FileManager::FileManager(string file_Name)
{
	fstream file;
	file.open(file_Name);
	string str_data;

	while (!file.eof())
	{
		getline(file, str_data, '\t');

	}
}
