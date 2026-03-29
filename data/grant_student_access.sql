-- SQL Script to Grant Student Access to nihe_stock_funds Database
-- Database: nihe_stock_funds
-- Server: 10.28.255.9
-- General student username: student_user
-- Read-only access for safety

-- Step 1: Create the student login (if it doesn't exist)
CREATE LOGIN [student_user] 
WITH PASSWORD = 'StudentPass2024!',
DEFAULT_DATABASE = [nihe_stock_funds],
CHECK_POLICY = ON,
CHECK_EXPIRATION = OFF;

-- Step 2: Create the user in the nihe_stock_funds database
USE [nihe_stock_funds];
GO

CREATE USER [student_user] 
FOR LOGIN [student_user];
GO

-- Step 3: Grant read-only access to the student
EXEC sp_addrolemember 'db_datareader', 'student_user';

-- Step 4: Verify permissions - Check what roles the user belongs to
SELECT 
    dp1.name AS DatabaseRoleName,
    dp2.name AS DatabaseUserName
FROM sys.database_role_members drm
JOIN sys.database_principals dp1 ON drm.role_principal_id = dp1.principal_id
JOIN sys.database_principals dp2 ON drm.member_principal_id = dp2.principal_id
WHERE dp2.name = 'student_user';

-- Step 5: Verify specific permissions
SELECT 
    perm.permission_name,
    perm.state_desc,
    obj.name AS object_name,
    user_name(perm.grantee_principal_id) AS grantee
FROM sys.database_permissions perm
JOIN sys.objects obj ON perm.major_id = obj.object_id
WHERE user_name(perm.grantee_principal_id) = 'student_user';

-- Step 6: Test connection and access - Run basic queries
-- Test 1: Check database access
SELECT DB_NAME() AS CurrentDatabase;

-- Test 2: Check what tables are accessible (read-only)
SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_SCHEMA, TABLE_NAME;

-- Test 3: Test data reading (example - replace with actual table name)
-- SELECT TOP 5 * FROM [your_table_name];

-- Step 7: Additional verification - Check user existence and permissions
-- Check if login exists at server level
SELECT name, type_desc, is_disabled
FROM sys.server_principals
WHERE name = 'student_user';

-- Check if user exists in database
SELECT name, type_desc, authentication_type_desc
FROM sys.database_principals
WHERE name = 'student_user';

-- List all permissions for the user
SELECT 
    perm.state_desc AS PermissionState,
    perm.permission_name AS PermissionName,
    obj.name AS ObjectName,
    obj.type_desc AS ObjectType
FROM sys.database_permissions perm
LEFT JOIN sys.objects obj ON perm.major_id = obj.object_id
WHERE perm.grantee_principal_id = USER_ID('student_user')
ORDER BY perm.permission_name, obj.name;