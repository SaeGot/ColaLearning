#include "FileManager.h"


FileManager::FileManager(string file_Name)
{
	fstream file;
	file.open(file_Name);

	int row = 0;
	string line;
	while (getline(file, line))
	{
		int col = 0;
		vector<string> line_data;
		stringstream ss_line;
		string str_data;
		while (getline(ss_line, str_data, '\t'))
		{
			//line_data.push_back(stod(str_data));
			line_data.push_back(str_data);
			col++;
		}
		data.push_back(line_data);
		row++;
	}
	// 컬럼명 설정
	columnName = data[0];
	// 첫 행(헤더) 제거
	data.erase(data.begin());
}
