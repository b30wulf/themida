#include "pch.h"

#pragma comment(lib, "xed.lib")
#pragma comment(lib, "triton.lib")

#include <fstream>
#include <TlHelp32.h>
#include <tchar.h>
#include <chrono>

#include "ProcessStream.hpp"
#include "ThemidaAnalyzer.hpp"
#include "CFG.hpp"


struct redirect_cout
{
	redirect_cout(const std::ostream &ostream)
	{
		this->m_backup = std::cout.rdbuf();
		std::cout.rdbuf(ostream.rdbuf());
	}

	~redirect_cout()
	{
		std::cout.rdbuf(this->m_backup);
	}

private:
	std::streambuf *m_backup;
};


DWORD find_process(LPCTSTR processName)
{
	DWORD processId = 0;
	HANDLE hProcessSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (hProcessSnap != INVALID_HANDLE_VALUE)
	{
		// Set the size of the structure before using it.
		PROCESSENTRY32 pe32;
		pe32.dwSize = sizeof(PROCESSENTRY32);
		if (Process32First(hProcessSnap, &pe32))
		{
			do
			{
				static size_t c_nLength = _tcslen(processName);
				size_t nLength = _tcslen(pe32.szExeFile);
				if (nLength < c_nLength ||
					_tcscmp(pe32.szExeFile + nLength - c_nLength, processName) != 0)
				{
					continue;
				}

				processId = pe32.th32ProcessID;
				break;
			} while (Process32Next(hProcessSnap, &pe32));
		}
		CloseHandle(hProcessSnap);
	}
	return processId;
}


void devirtualizeme32(LPCTSTR processName, int argc, char* argv[])
{
	DWORD processId = find_process(processName);
	printf("pid: %08X\n", processId);

	ProcessStream stream(false);
	if (!stream.open(processId))
		throw std::runtime_error("stream.open failed.");

	// get address from arg
	auto start = std::chrono::steady_clock::now();
	triton::uint64 start_address;
	if (argc == 2)
	{
		start_address = std::strtol(argv[1], nullptr, 16);
	}
	else
	{
		// fish32 white start
		start_address = 0x0040CA4A; // fish32 white
		//start_address = 0x0040C97A; // fish32 red
		//start_address = 0x0040C89A; // fish32 black

		// tiger32
		//start_address = 0x0040CA5A; // tiger32 white
		//start_address = 0x9ba0d9; // tiger32 white next
		//start_address = 0x0040C97A; // tiger32 red
		//start_address = 0x0040C89A; // tiger32 black

		// dolphin32
		//start_address = 0x0040CA6A; // dolphin32 white
	}

	// redirect cout to file
	char filename[1024];
	//sprintf_s(filename, 1024, "D:\\virtualmachine\\devirtualizeme\\%ls-%llX.txt", processName, start_address);
	sprintf_s(filename, 1024, "%ls-%llX.txt", processName, start_address);
	std::ofstream os(filename);
	redirect_cout _r(os);

	// work
	ThemidaAnalyzer tmd_analyzer(triton::arch::ARCH_X86);
	try
	{
		tmd_analyzer.analyze(stream, start_address);
	}
	catch (const std::exception& ex)
	{
		printf("ex: %s\n", ex.what());
	}
	tmd_analyzer.print_output();

	// print execution time
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	const double diff_ms = std::chrono::duration<double, std::milli>(diff).count();
	printf("%f mseconds\n", diff_ms);
}


void devirtualizeme64(LPCTSTR processName, int argc, char* argv[])
{
	DWORD processId = find_process(processName);
	printf("pid: %08X\n", processId);

	ProcessStream stream(true);
	if (!stream.open(processId))
		throw std::runtime_error("stream.open failed.");

	// get address from arg
	auto start = std::chrono::steady_clock::now();
	ThemidaAnalyzer analyzer(triton::arch::ARCH_X86_64);
	triton::uint64 start_address = 0x009BA537;
	if (argc == 2)
	{
		start_address = std::strtol(argv[1], nullptr, 16);
	}
	else
	{
		// tiger64 white start
		start_address = 0x140001D37ull;
		//start_address = 0x140942C82ull; // last address so easy to test something quick

		// fish64 white
		//start_address = 0x140001D37ull;
	}

	// redirect cout to file
	char filename[1024];
	sprintf_s(filename, 1024, "D:\\virtualmachine\\devirtualizeme\\%ls-%llX.txt", processName, start_address);
	sprintf_s(filename, 1024, "%ls-%llX.txt", processName, start_address);
	std::ofstream os(filename);
	redirect_cout _r(os);

	// work
	try
	{
		analyzer.analyze(stream, start_address);
		analyzer.print_output();
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
		analyzer.print_output();
	}

	// print execution time
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	const double diff_ms = std::chrono::duration<double, std::milli>(diff).count();
	printf("%f mseconds\n", diff_ms);
}


int main(int argc, char *argv[])
{
	// Once, before using Intel XED, you must call xed_tables_init() to initialize the tables Intel XED uses for encoding and decoding:
	xed_tables_init();

	try
	{
		devirtualizeme32(L"devirtualizeme_tmd_2.4.6.0_fish32.exe", argc, argv);
		//devirtualizeme32(L"devirtualizeme_tmd_2.4.6.0_tiger32.exe", argc, argv);
		//devirtualizeme32(L"devirtualizeme_tmd_2.4.6.0_dolphin32.exe", argc, argv);

		//devirtualizeme64(L"devirtualizeme_tmd_2.4.6.0_tiger64.exe", argc, argv);
		//devirtualizeme64(L"devirtualizeme_tmd_2.4.6.0_fish64.exe", argc, argv);
	}
	catch (const triton::exceptions::Exception& ex)
	{
		std::cout << "trion ex: " << ex.what() << std::endl;
	}
	catch (const std::exception &ex)
	{
		std::cout << ex.what() << std::endl;
	}
	return 0;
}