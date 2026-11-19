package com.aifinancialadvisor.user.dto;

import java.util.UUID;
import com.aifinancialadvisor.user.Role;
import com.aifinancialadvisor.user.User;

public record UserResponse(
        UUID id,
        String email,
        Role role,
        boolean verified) {

    public static UserResponse from(User user) {
        return new UserResponse(
                user.getId(),
                user.getEmail(),
                user.getRole(),
                user.isVerified());
    }
}