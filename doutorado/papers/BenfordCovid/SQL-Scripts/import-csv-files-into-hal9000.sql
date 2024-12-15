-- 1. Optional: create the destination table if it does not already exist.
--    Adjust the column definitions to match your CSV structure.
drop table [BigData].[dbo].[COVID-19-DAILY-WHO]

CREATE TABLE [BigData].[dbo].[COVID-19-DAILY-WHO](
	[Date_reported] [nvarchar](50) NULL,
	[Country_code] [nvarchar](50) NULL,
	[Country] [nvarchar](250) NULL,
	[WHO_region] [nvarchar](50) NULL,
	[New_cases] [nvarchar](50) NULL,
	[Cumulative_cases] [nvarchar](50) NULL,
	[New_deaths] [nvarchar](50) NULL,
	[Cumulative_deaths] [nvarchar](50) NULL
) ON [PRIMARY]

-- 2. Use the BULK INSERT command to read data from the CSV and load it into the table.
--    Update the path to reflect your actual CSV file location.
--    If your CSV has a header row, use FIRSTROW = 2 to skip it.
--    Adjust FIELDTERMINATOR and ROWTERMINATOR as needed.
BULK INSERT [BigData].[dbo].[COVID-19-DAILY-WHO]
FROM 'C:\Users\cazev\OneDrive\docs\Doutorado\papers\BenfordCovid\datasources\WHO-COVID-19-global-daily-data.csv'
WITH
(
    FIRSTROW = 2,              -- If the first row of the CSV is headers, start from the 2nd row.
    FIELDTERMINATOR = ',',     -- CSV fields are typically comma-separated; adjust if needed.
    ROWTERMINATOR = '\n',      -- Usually each new line represents a new row.
    TABLOCK
);

-- 3. Check the imported data
SELECT TOP 10 *
FROM [BigData].[dbo].[COVID-19-DAILY-WHO]
