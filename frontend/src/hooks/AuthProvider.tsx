/**
 * MedRittAI — Auth Provider
 * Provides authentication state and login/logout actions.
 */

import React, { useState, useCallback, useEffect } from 'react';
import { getMe, login as apiLogin, register as apiRegister, logout as apiLogout, setAuthToken } from '../api/client';
import type { LoginRequest, RegisterRequest, UserSummary } from '../types';
import { AuthContext } from './authContext';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('medrittai_token')
  );
  const [username, setUsername] = useState<string | null>(
    localStorage.getItem('medrittai_username')
  );
  const [user, setUser] = useState<UserSummary | null>(() => {
    const saved = localStorage.getItem('medrittai_user');
    try { return saved ? JSON.parse(saved) : null; } catch { return null; }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (token) {
      setAuthToken(token);
      if (!user) {
        getMe().then((identity) => {
          setUser(identity);
          setUsername(identity.username);
          localStorage.setItem('medrittai_user', JSON.stringify(identity));
          localStorage.setItem('medrittai_username', identity.username);
        }).catch(() => {
          apiLogout();
          setToken(null);
        });
      }
    }
  }, [token, user]);

  const login = useCallback(async (data: LoginRequest) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiLogin(data);
      setToken(res.access_token);
      setUsername(data.username);
      setUser(res.user);
      localStorage.setItem('medrittai_username', data.username);
      localStorage.setItem('medrittai_user', JSON.stringify(res.user));
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Login failed. Please try again.';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const register = useCallback(async (data: RegisterRequest) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiRegister(data);
      setToken(res.access_token);
      setUsername(res.user.username);
      setUser(res.user);
      localStorage.setItem('medrittai_username', res.user.username);
      localStorage.setItem('medrittai_user', JSON.stringify(res.user));
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Registration failed. Please try again.';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    apiLogout();
    setToken(null);
    setUsername(null);
    setUser(null);
    localStorage.removeItem('medrittai_username');
    localStorage.removeItem('medrittai_user');
  }, []);

  const setCurrentUser = useCallback((identity: UserSummary) => {
    setUser(identity);
    setUsername(identity.username);
    localStorage.setItem('medrittai_user', JSON.stringify(identity));
    localStorage.setItem('medrittai_username', identity.username);
  }, []);

  const refreshUser = useCallback(async () => {
    const identity = await getMe();
    setCurrentUser(identity);
    return identity;
  }, [setCurrentUser]);

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated: !!token,
        username,
        user,
        token,
        login,
        register,
        logout,
        refreshUser,
        setCurrentUser,
        loading,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
