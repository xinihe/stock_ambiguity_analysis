# DBeaver Guide: Granting Student Access to SQL Server Database

## Overview
This guide explains how to use the `grant_student_access.sql` script in DBeaver to create a student account with read-only access to your `nihe_stock_funds` database.

## Student Account Details
- **Username**: `student_user`
- **Password**: `StudentPass2024!`
- **Access Level**: Read-only (`db_datareader` role)
- **Database**: `nihe_stock_funds`
- **Server**: `10.28.255.9`

## Step-by-Step Instructions

### Step 1: Connect to Your SQL Server in DBeaver

1. **Open DBeaver**
2. **Create a new connection** (if not already connected):
   - Click "New Database Connection" or press `Ctrl+Shift+N`
   - Select "SQL Server" from the list
   - Click "Next"
3. **Configure connection settings**:
   - **Host**: `10.28.255.9`
   - **Port**: `1433` (default)
   - **Database**: `nihe_stock_funds`
   - **Authentication**: SQL Server Authentication
   - **Username**: Your admin username
   - **Password**: Your admin password
4. **Test Connection** and click "Finish"

### Step 2: Execute the SQL Script

1. **Open SQL Editor**:
   - Right-click on your connection → "SQL Editor" → "Open SQL Script"
   - Or press `Ctrl+Alt+Shift+S`

2. **Load the SQL Script**:
   - File → Open File → Navigate to `/Users/tlxy/Research/Ambiguity/data/grant_student_access.sql`
   - Or copy and paste the script content

3. **Execute the Script**:
   - Select all SQL commands (`Ctrl+A`)
   - Press `Ctrl+Enter` or click the "Execute SQL Script" button
   - **Important**: Execute each section separately if you encounter errors:
     - First, execute the `CREATE LOGIN` statement
     - Then execute the `USE [nihe_stock_funds]` and `CREATE USER` statements
     - Finally, execute the permission and verification queries

### Step 3: Verify the Student Account

After executing the script, you should see results similar to:

#### Verification Results:
```
DatabaseRoleName | DatabaseUserName
-----------------|------------------
db_datareader    | student_user
```

#### Connection Test Results:
```
CurrentDatabase
---------------
nihe_stock_funds
```

#### Table Access Test:
You should see a list of all tables in your database that the student can read.

### Step 4: Test Student Connection in DBeaver

1. **Create a new connection for testing**:
   - File → New → Database Connection
   - Select "SQL Server"

2. **Use student credentials**:
   - **Host**: `10.28.255.9`
   - **Port**: `1433`
   - **Database**: `nihe_stock_funds`
   - **Authentication**: SQL Server Authentication
   - **Username**: `student_user`
   - **Password**: `StudentPass2024!`

3. **Test the connection** and click "Finish"

4. **Verify read-only access**:
   - Expand the connection → Schemas → dbo → Tables
   - Right-click on any table → "View Data"
   - Try to edit data (should fail with read-only permissions)

## DBeaver-Specific Features

### Using SQL Editor for Verification

After creating the student account, you can run these verification queries in DBeaver:

```sql
-- Check if student_user exists
SELECT name, type_desc, create_date
FROM sys.server_principals
WHERE name = 'student_user';

-- Check database permissions
SELECT 
    dp1.name AS RoleName,
    dp2.name AS UserName
FROM sys.database_role_members drm
JOIN sys.database_principals dp1 ON drm.role_principal_id = dp1.principal_id
JOIN sys.database_principals dp2 ON drm.member_principal_id = dp2.principal_id
WHERE dp2.name = 'student_user';
```

### Connection Management

1. **Save the student connection**:
   - Right-click on the connection → "Connection Settings"
   - Check "Save password locally" (optional)
   - Click "OK"

2. **Share connection settings**:
   - Right-click on the connection → "Connection Settings"
   - Click "Export" to save connection configuration
   - Share the exported file with students

## Troubleshooting in DBeaver

### Common Issues:

1. **"Login failed for user"**:
   - Verify username: `student_user`
   - Verify password: `StudentPass2024!`
   - Check if SQL Server allows SQL Server authentication

2. **"Cannot open database"**:
   - Verify database name: `nihe_stock_funds`
   - Check if user has proper permissions
   - Run verification queries to check user mapping

3. **Connection timeout**:
   - Check network connectivity to `10.28.255.9`
   - Verify SQL Server is running
   - Check firewall settings for port 1433

### DBeaver-Specific Solutions:

1. **Reset connection**:
   - Right-click connection → "Invalidate/Reconnect"

2. **Check driver version**:
   - Help → "About DBeaver" → "Installation Information"
   - Update SQL Server driver if needed

3. **Enable SQL logging**:
   - Window → "Preferences" → "General" → "Logging"
   - Enable SQL execution logging for debugging

## Security Notes

- **Change the default password** before giving access to students
- **Monitor student connections** using SQL Server logs
- **Consider time-based access** if students only need temporary access
- **Regular review** of student permissions and remove when no longer needed

## Quick Reference

**Student Connection Details:**
- Server: `10.28.255.9`
- Database: `nihe_stock_funds`
- Username: `student_user`
- Password: `StudentPass2024!`
- Port: `1433`

**Key DBeaver Shortcuts:**
- Open SQL Editor: `Ctrl+Alt+Shift+S`
- Execute SQL: `Ctrl+Enter`
- New Connection: `Ctrl+Shift+N`
- View Table Data: Right-click → "View Data"