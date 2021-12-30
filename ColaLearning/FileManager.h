#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>


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
	 * \param data_Types
	 */
	FileManager(string file_Name, vector<Type> data_Types);
	~FileManager();

private:
	vector<string> columnName;
	vector<Type> dataType;
	vector<vector<double>> data;

	/**
	 * 컬럼명 설정.
	 * 
	 * \param line : 첫 행
	 */
	void SetColumnName(string first_line);
	/**
	 * 모든 타입을 동일하게 설정.
	 * 
	 * \param data_Type : 데이터 타입
	 */
	void SetDataType(Type data_Type);
	/**
	 * 각 칼럼의 타입을 순서대로 설정.
	 * 
	 * \param data_Types
	 */
	void SetDataType(vector<Type> data_Types);
	/**
	 * 데이터 설정.
	 * 
	 * \param line : 행
	 */
	void SetData(string line);
};

