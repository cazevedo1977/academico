/*
Script responsavel pela inserção e tratamento de dados para plataforma HAL 9000, para computar os indicadores necessários para análise dos dados 
baseados no método proposto da detecção de fraudes.

Caio Azevedo - 08.12.24

/* 
PASSO 1: INSERE TODOS OS REGISTROS PROVENIENTES DA TABELA TEMPORARIA NA TABELA DO SISTEMA
delete [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM] where BFAM_AMOSTRA_TIPO = 31

insere os dados diários nas amostras, 30 new_cases - 31 new_deaths
insert into [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]

SELECT 
    30,
    ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS id,
    country + ' - ' + who_region + ' - ' + date_reported ,
    new_cases,
    WHO_region,
    DENSE_RANK() OVER (ORDER BY country) AS country_code,
    Country_code
FROM [BigData].[dbo].[COVID-19-DAILY-WHO]
*/

/*
PASSO 2: REMOVE OS REGISTOS COM O VALOR NULO OU ZERADO

DECLARE @amostra_tipo INT,
		@tot INT, 
        @tot_null INT;

SELECT @amostra_tipo = 31
-- Atribui a quantidade total
SELECT @tot = COUNT(*) 
FROM [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]
WHERE BFAM_AMOSTRA_TIPO = @amostra_tipo;

-- Atribui a quantidade total nula ou zero
SELECT @tot_null = COUNT(*) 
FROM [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]
WHERE BFAM_AMOSTRA_TIPO = @amostra_tipo
  AND (BFAM_VALOR IS NULL OR BFAM_VALOR = '0');

-- Exibe as variáveis e o resultado da divisão
SELECT @tot AS Total, 
       @tot_null AS TotalNulo,
       CASE WHEN @tot_null = 0 THEN NULL ELSE @tot_null * 1.0 / @tot END AS ResultadoDivisao;

DELETE [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]
WHERE BFAM_AMOSTRA_TIPO = @amostra_tipo
  AND (BFAM_VALOR IS NULL OR BFAM_VALOR = '0');

*/

/*
PASSO 3: CONTA TOTAL DE REGISTROS ANALISAVEIS

DECLARE @amostra_tipo INT
SELECT @amostra_tipo = 30
-- Atribui a quantidade total
SELECT COUNT(*) 
FROM [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]
WHERE BFAM_AMOSTRA_TIPO = @amostra_tipo;
*/

/*
PASSO 4: VALIDA QUANTIDADE DE REGISTROS PRONTOS PARA ANÁLISE - CRUZANDO COM EDA - PYTHON

select count(*) from [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]
where BFAM_AMOSTRA_TIPO = 30
AND BFAM_DESCRICAO LIKE 'china%'

*/

/*
BONUS ROUND - IDENTIFICA O ID CATEGORIA DE CADA PAIS E VERIFICA SE OS PROCESSAMENTOS FORAM FEITOS 

SELECT DISTINCT BFAM_AMOSTRA_TIPO,BFAM_CATEGORIA,BFAM_CATEGORIA_CH FROM [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]
WHERE BFAM_AMOSTRA_TIPO = 31 ORDER BY BFAM_CATEGORIA_CH

DELETE [BigData].[dbo].[TB_BENFORD_RESULT_BFRE] WHERE BFRE_AMOSTRA_TIPO = 31 AND BFRE_CATEGORIA = 210
DELETE [BigData].[dbo].[TB_BENFORD_TESTES_BFTT] WHERE BFTT_AMOSTRA_ID = 31 AND BFTT_CATEGORIA = 20

select * from [BigData].[dbo].[TB_BENFORD_RESULT_BFRE]
WHERE BFRE_AMOSTRA_TIPO = 31 AND BFRE_CATEGORIA = 74 order by BFRE_NUMERO

select * from [BigData].[dbo].[TB_BENFORD_TESTES_BFTT]
WHERE BFTT_AMOSTRA_ID = 31 and BFTT_CATEGORIA = 74 order by BFTT_NUMERO

*/

-- select top 10 * from [BigData].[dbo].[COVID-19-DAILY-WHO]
-- select count(*) from [BigData].[dbo].[COVID-19-DAILY-WHO]
--select * from delete [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM] where BFAM_AMOSTRA_TIPO = 30
-- select * from delete [BigData].[dbo].[TB_BENFORD_RESULT_BFRE] where BFRE_AMOSTRA_TIPO = 30
-- select * from [BigData].[dbo].[TB_BENFORD_TESTES_BFTT] where BFTT_AMOSTRA_ID = 30



/* VERIFICA OS paises com maior quantidade de dados informados
select distinct Country, Country_code from [BigData].[dbo].[COVID-19-DAILY-WHO] order by Country

select BFAM_CATEGORIA_CH,count(*) as  total from [BigData].[dbo].[TB_BENFORD_AMOSTRA_BFAM]
where BFAM_AMOSTRA_TIPO = 30
group by BFAM_CATEGORIA_CH
order by total desc
*/









