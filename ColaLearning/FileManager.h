#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>


using namespace std;

class FileManager
{
public:
	enum class Type
	{
		Real,
		String
	};
	/**
	 * 모든 타입을 동일하게 데이터 생성.
	 *
	 * \param file_Name : 파일 이름
	 * \param data_Type : 데이터 타입
	 */
	FileManager(string file_Name, Type data_Type = Type::Real);
	/**
	 * 각각의 칼럼에 타입을 설정하여 데이터 생성.
	 * 
	 * \param file_Name
	 * \param data_TypeList
	 */
	FileManager(string file_Name, vector<Type> data_Types);
	~FileManager();
	/**
	 * 해당 행렬 데이터 가져오기.
	 * 
	 * \param row : 행
	 * \param column : 열
	 * \return 데이터
	 */
	double GetData(int row, int column);
	/**
	 * 해당 행 데이터 가져오기.
	 * 
	 * \param row : 행
	 * \return 행 데이터
	 */
	vector<double> GetData(int row);


private:
	struct Data
	{
		string columnName;
		Type type;
		// <행, 값>
		map<int, double> value;
	};
	// <열, 값>
	map<int, Data> data;

	/**
	 * 칼럼명 설정.
	 * 
	 * \param line : 첫 행
	 * \return 칼럼명
	 */
	vector<string> SetColumnName(string first_line);
	/**
	 * 모든 타입을 동일하게 설정.
	 * 
	 * \param column_Names : 칼럼명
	 * \param data_Type : 데이터 타입
	 */
	vector<Type> SetDataType(vector<string> column_Names, Type data_Type);
	/**
	 * 각 칼럼의 타입을 순서대로 설정.
	 * 
	 * \param column_Names : 칼럼명
	 * \param data_TypeList : 데이터 타입
	 */
	vector<Type> SetDataType(vector<string> column_Names, vector<Type> data_Types);
	/**
	 * 데이터 설정.
	 * 
	 * \param row : 행 인덱스
	 * \param line : 행 데이터
	 */
	map<int, double> SetData(int row, string line);
};

